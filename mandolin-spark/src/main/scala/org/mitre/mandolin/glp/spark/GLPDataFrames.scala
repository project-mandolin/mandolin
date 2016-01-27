package org.mitre.mandolin.glp.spark

import org.mitre.mandolin.config.{LearnerSettings, OnlineLearnerSettings}

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.AccumulatorParam

import org.mitre.mandolin.glp._
import org.mitre.mandolin.optimize.spark.{DistributedOnlineOptimizer, DistributedOptimizerEstimator}
import org.mitre.mandolin.util.{Alphabet, IdentityAlphabet, DenseTensor1 => DenseVec, IOAssistant}
import org.mitre.mandolin.util.spark.SparkIOAssistant

/**
 * Utilities for mapping GLP factors to Spark DataFrames
 * @author wellner
 */
trait GLPDataFrames {
  import org.apache.spark.sql.types.{StructType,StructField,StringType, DoubleType, IntegerType}

  //import org.apache.spark.mllib.linalg.VectorUDT
  import org.apache.spark.mllib.linalg.{VectorUDT, Vectors}
  import org.apache.spark.sql.DataFrame
  
  private def toDouble(x: Array[Float]) = x map {_.toDouble}
  
  def mapGLPFactorsToDf(sqlSc: SQLContext, fvs: RDD[GLPFactor], dim: Int) = {
    val im = (0 to dim).toArray
    val rows = fvs map {f =>
      val label = f.getOneHot.toDouble // label as the one-hot output index
      val fv = f.getInput
      val rseq = (label, Vectors.dense(toDouble(fv.asArray)))
      org.apache.spark.sql.Row.fromTuple(rseq)
      }
    val schema = StructType(Seq(StructField("label",DoubleType,true), StructField("features", new org.apache.spark.mllib.linalg.VectorUDT, true))) 
    val df = sqlSc.createDataFrame(rows,schema)
    df
  }
  
  def mapGLPFactorsToDf(sqlSc: SQLContext, fvs: RDD[GLPFactor], alphabet: Alphabet) = {
    val im1 = alphabet.getInverseMapping.toList.sortWith { case ((k1, v1), (k2, v2)) => k1 < k2 }
    val im = (0,"label") :: im1
    val ar = im.toArray
    val rows = fvs map {f =>
      val label = f.getOneHot.toDouble // label as the one-hot output index
      val fv = f.getInput
      val rseq = (label, Vectors.dense(toDouble(fv.asArray)))
      org.apache.spark.sql.Row.fromTuple(rseq)
      }
    val schema = StructType(Seq(StructField("label",DoubleType,true), StructField("features", new org.apache.spark.mllib.linalg.VectorUDT, true)))
    val df = sqlSc.createDataFrame(rows, schema)
    df
  }

  /**
   * Map a Spark DataFrame back into an `RDD` of `GLPFactor` objects.  Assumes the input
   * DataFrame just has two columns (label, features)
   */
  def mapDfToGLPFactors(sc: SparkContext, df: DataFrame, fvDim: Int, labelDim: Int) = {
    val rows : RDD[GLPFactor] = df map {row =>
      val outVec = Array.fill(labelDim)(0.0)
      val r = row.getDouble(0).toInt
      outVec(r) = 1.0
      val vec = row.getAs[org.apache.spark.mllib.linalg.Vector](1) 
      val inVec = vec.toArray
      new StdGLPFactor(-1, new DenseVec(inVec), new DenseVec(outVec), None)
      }
    rows
  }
}

class GlpModel extends GLPDataFrames with Serializable {
  import org.apache.spark.sql.DataFrame
  import org.apache.spark.sql._
  import org.apache.spark.sql.functions._
  import org.mitre.mandolin.optimize.Updater
  import scala.reflect.ClassTag
    
  def copy() = new GlpModel
  
  def readAsDataFrame(sqlContext: SQLContext, sc: SparkContext, fPath: String, idim: Int, odim: Int) = {
    val lines = sc.textFile(fPath)
    val dp = new DistributedProcessor()
    val fe = new StdVectorExtractorWithAlphabet(new IdentityAlphabet(odim, false), new IdentityAlphabet(idim), idim)
    val fvs = lines map {fe.extractFeatures}
    mapGLPFactorsToDf(sqlContext, fvs, idim)
  }
  
  def evaluate(mspec: GLPModelSpec, tstdata: DataFrame) : org.mitre.mandolin.predict.DiscreteConfusion = {
    val sc = tstdata.sqlContext.sparkContext
    val dp = new DistributedProcessor()
    val dim = mspec.wts.getInputDim
    val odim = mspec.wts.getOutputDim
    val predictor = new GLPPredictor(mspec.ann, true)
    val evalDecoder = new org.mitre.mandolin.predict.spark.EvalDecoder(mspec.fe, predictor)
    val wBc = sc.broadcast(mspec.wts)
    val tstVecs = mapDfToGLPFactors(sc, tstdata, dim, odim)
    evalDecoder.evalUnits(tstVecs, wBc)
  }
  
  def estimate(trdata: DataFrame, modelSpec: IndexedSeq[LType]) : GLPModelSpec = {
    if (modelSpec.length < 3) estimate(trdata, modelSpec, AdaGradSpec(0.1))
    else estimate(trdata, modelSpec, RMSPropSpec(0.001))
  }
  
  def estimate(trdata: DataFrame, mSpec: IndexedSeq[LType], upSpec: UpdaterSpec, 
      epochs: Int = 20, threads: Int = 8, partitions: Int = 0) : GLPModelSpec = {
    val dp = new DistributedProcessor()
    val idim = trdata.select("features").head().get(0).asInstanceOf[org.apache.spark.mllib.linalg.Vector].size 
    val odim = trdata.select("label").rdd map {v => v.getDouble(0)} max // get the maximum value of the label column 
    val modelSpec = ANNetwork.fullySpecifySpec(mSpec, idim, odim.toInt + 1)
    val components = dp.getComponentsDenseVecs(modelSpec)
    val glp = components.ann
    val layout = glp.generateZeroedLayout
    upSpec match {
      case AdaGradSpec(lr)     => estimate(components, trdata, modelSpec, new GLPAdaGradUpdater(layout, lr.toFloat), epochs, threads, partitions)
      case RMSPropSpec(lr)     => estimate(components, trdata, modelSpec, new GLPRMSPropUpdater(layout, lr.toFloat), epochs, threads, partitions)
      case SgdSpec(lr)         => estimate(components, trdata, modelSpec, new GLPSgdUpdater(layout, false, lr.toFloat), epochs, threads, partitions)
      case AdaDeltaSpec        => estimate(components, trdata, modelSpec, new GLPAdaDeltaUpdater(layout, layout.copy()), epochs, threads, partitions)
      case NesterovSgdSpec(lr) =>
        val np = trdata.count().toInt // get total number of training points to scale momentum in Nesterov accelerated SGD
        estimate(components, trdata, modelSpec, new GLPSgdUpdater(layout, true, lr.toFloat, numPoints = np), epochs, threads, partitions)
    }    
  }
  
  def estimate[U <: Updater[GLPWeights, GLPLossGradient, U]: ClassTag](components: GLPComponentSet, trdata: DataFrame, 
      modelSpec: IndexedSeq[LType], updater: U, epochs: Int, threads: Int, numPartitions: Int) : GLPModelSpec = {
    val sc = trdata.sqlContext.sparkContext
    val dim = components.dim
    val odim = components.labelAlphabet.getSize
    val glp = components.ann
    val ev = new GLPInstanceEvaluator[U](glp)
    val optimizer = 
      new DistributedOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, U](sc, glp.generateRandomWeights, ev, updater, 
          epochs, 1, threads, None)
    val fvs1 = mapDfToGLPFactors(sc, trdata, dim, odim)
    val fvs = if (numPartitions > 0) fvs1.repartition(numPartitions) else fvs1
    fvs.persist()
    val (w,_) = optimizer.estimate(fvs)
    GLPModelSpec(w, glp, components.labelAlphabet, components.featureExtractor)    
  }
  
  def estimate(trdata: DataFrame, appSettings: GLPModelSettings) : GLPModelSpec = {
    val dp = new DistributedProcessor(appSettings.numPartitions)
    val io = new SparkIOAssistant(trdata.sqlContext.sparkContext)
    val components = dp.getComponentsViaSettings(appSettings, io)       
    val fe = components.featureExtractor
    val trainFile = appSettings.trainFile
    val sc = AppConfig.getSparkContext(appSettings)
    val numPartitions = appSettings.numPartitions
    val network = components.ann
    val optimizer: DistributedOptimizerEstimator[GLPFactor, GLPWeights] = DistributedGLPOptimizer.getDistributedOptimizer(sc, appSettings, network)
    val fvs1 = mapDfToGLPFactors(sc, trdata, trdata.columns.length - 1, components.labelAlphabet.getSize)
    val fvs = if (numPartitions > 0) fvs1.repartition(numPartitions) else fvs1    
    fvs.persist()
    val (w, _) = optimizer.estimate(fvs, Some(appSettings.numEpochs))
    GLPModelSpec(w, network, components.labelAlphabet, components.featureExtractor)
  }
  
}

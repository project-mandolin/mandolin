package org.mitre.mandolin.glp.spark

import org.mitre.mandolin.config.{LearnerSettings, OnlineLearnerSettings}

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

import org.mitre.mandolin.glp._
import org.mitre.mandolin.optimize.spark.{DistributedOnlineOptimizer, DistributedOptimizerEstimator}
import org.mitre.mandolin.util.{Alphabet, IdentityAlphabet, DenseTensor1 => DenseVec, IOAssistant}

/**
 * Utilities for mapping GLP factors to Spark DataFrames
 * @author wellner
 */
trait GLPDataFrames {
  import org.apache.spark.sql.types.{StructType,StructField,StringType, DoubleType, IntegerType}

  //import org.apache.spark.mllib.linalg.VectorUDT
  import org.apache.spark.mllib.linalg.{VectorUDT, Vectors}
  import org.apache.spark.sql.DataFrame
  
  def mapGLPFactorsToDf(sqlSc: SQLContext, fvs: RDD[GLPFactor], dim: Int) = {
    val im = (0 to dim).toArray
    val rows = fvs map {f =>
      val label = f.getOneHot.toDouble // label as the one-hot output index
      val fv = f.getInput
      val rseq = (label, Vectors.dense(fv.asArray))
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
      val rseq = (label, Vectors.dense(fv.asArray))
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

class GlpModel extends GLPDataFrames {
  import org.apache.spark.sql.DataFrame
  import org.apache.spark.sql._
  import org.apache.spark.sql.functions._
  import org.mitre.mandolin.optimize.Updater
  import scala.reflect.ClassTag
    
  def copy() = new GlpModel
  
  def readAsDataFrame(sqlContext: SQLContext, sc: SparkContext, fPath: String, idim: Int, odim: Int) = {
    val lines = sc.textFile(fPath)
    val dp = new DistributedProcessor()
    val fe = new StdVectorExtractorWithAlphabet(new IdentityAlphabet(odim), idim)
    val fvs = lines map {fe.extractFeatures}
    mapGLPFactorsToDf(sqlContext, fvs, idim)
  }
  
  def evaluate(mspec: GLPModelSpec, tstdata: DataFrame) : org.mitre.mandolin.predict.DiscreteConfusion = {
    val sc = tstdata.sqlContext.sparkContext
    val dp = new DistributedProcessor()
    val dim = mspec.wts.getInputDim
    val odim = mspec.wts.getOutputDim
    val predictor = new GLPPredictor(mspec.evaluator.glp, true)
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
      epochs: Int = 20, threads: Int = 8) : GLPModelSpec = {
    val dp = new DistributedProcessor()
    val idim = trdata.select("features").head().get(0).asInstanceOf[org.apache.spark.mllib.linalg.Vector].size 
    val odim = trdata.select("label").rdd map {v => v.getDouble(0)} max // get the maximum value of the label column 
    val modelSpec = ANNetwork.fullySpecifySpec(mSpec, idim, odim.toInt + 1)
    val components = dp.getComponentsDenseVecs(modelSpec)
    val glp = components.evaluator.glp
    val layout = glp.generateZeroedLayout
    upSpec match {
      case AdaGradSpec(lr)     => estimate(components, trdata, modelSpec, new GLPAdaGradUpdater(layout, lr), epochs, threads)
      case RMSPropSpec(lr)     => estimate(components, trdata, modelSpec, new GLPRMSPropUpdater(layout, lr), epochs, threads)
      case SgdSpec(lr)         => estimate(components, trdata, modelSpec, new GLPSgdUpdater(layout, false, lr), epochs, threads)
      case AdaDeltaSpec        => estimate(components, trdata, modelSpec, new GLPAdaDeltaUpdater(layout, layout.copy()), epochs, threads)
      case NesterovSgdSpec(lr) =>
        val np = trdata.count().toInt // get total number of training points to scale momentum in Nesterov accelerated SGD
        estimate(components, trdata, modelSpec, new GLPSgdUpdater(layout, true, lr, numPoints = np), epochs, threads)
    }    
  }
  
  def estimate[U <: Updater[GLPWeights, GLPLossGradient, U]: ClassTag](components: GLPComponentSet, trdata: DataFrame, 
      modelSpec: IndexedSeq[LType], updater: U, epochs: Int, threads: Int) : GLPModelSpec = {
    val sc = trdata.sqlContext.sparkContext
    val dim = components.dim
    val odim = components.labelAlphabet.getSize
    val glp = components.evaluator.glp
    val optimizer = 
      new DistributedOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, U](sc, glp.generateRandomWeights, components.evaluator, updater, 
          epochs, 1, threads, None)
    val fvs = mapDfToGLPFactors(sc, trdata, dim, odim)
    val (w,_) = optimizer.estimate(fvs)
    GLPModelSpec(w, components.evaluator, components.labelAlphabet, components.featureExtractor)    
  }
  
  def estimate(trdata: DataFrame, appSettings: GLPModelSettings) : GLPModelSpec = {
    val dp = new DistributedProcessor(appSettings.numPartitions)
    val io = new IOAssistant
    val components = dp.getComponentsViaSettings(appSettings, io)       
    val ev = components.evaluator
    val fe = components.featureExtractor
    val trainFile = appSettings.trainFile
    val sc = AppConfig.getSparkContext(appSettings)
    val network = ev.glp
    val optimizer: DistributedOptimizerEstimator[GLPFactor, GLPWeights] = DistributedGLPOptimizer.getDistributedOptimizer(sc, appSettings, network, ev)
    val fvs = mapDfToGLPFactors(sc, trdata, trdata.columns.length - 1, components.labelAlphabet.getSize)
    val (w, _) = optimizer.estimate(fvs, Some(appSettings.numEpochs))
    GLPModelSpec(w, components.evaluator, components.labelAlphabet, components.featureExtractor)
  }
  
}

package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.config.{AppConfig,LearnerSettings, OnlineLearnerSettings}
import org.mitre.mandolin.optimize.spark.{DistributedOnlineOptimizer, DistributedOptimizerEstimator}
import org.mitre.mandolin.optimize.local.LocalOnlineOptimizer
import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.util.{Alphabet, AlphabetWithUnitScaling, StdAlphabet, IdentityAlphabet, IOAssistant}
import org.mitre.mandolin.predict.spark.Trainer
import org.mitre.mandolin.gm.Feature
import org.mitre.mandolin.util.{LineParser, DenseTensor1 => DenseVec, SparseTensor1 => SparseVec, Tensor1}
import org.mitre.mandolin.glp.spark.DistributedProcessor

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

class GLPSettings(a: Seq[String]) extends LearnerSettings(a) with OnlineLearnerSettings 

/**
 * Maps input lines in a sparse-vector format `<label> <f1>:<val> <f2>:<val> ... `
 * to a dense vector input. This is most efficient if the inputs aren't ''too'' sparse
 * and the dimensionality isn't too great.
 * @param alphabet Feature alphabet
 * @param la Label alphabet
 * @author wellner
 */
class VecFeatureExtractor(alphabet: Alphabet, la: Alphabet)
  extends FeatureExtractor[String, GLPFactor] with LineParser with Serializable {
  var ind = 0
  
  def getAlphabet = alphabet
  
  def extractFeatures(s: String): GLPFactor = {
    val (l, spv, id) = sparseOfLine(s, alphabet, addBias = false)
    val dVec : DenseVec = DenseVec.zeros(alphabet.getSize)    
    spv foreach { f =>
      if (f.fid >= 0) {
        val fv = alphabet.getValue(f.fid, f.value)
        dVec.update(f.fid, fv)
      }
    }
    val l_ind = la.ofString(l)
    val lv = DenseVec.zeros(la.getSize)
    lv.update(l_ind,1.0) // one-hot encoding
    ind += 1
    new StdGLPFactor(ind, dVec, lv, id)    
  }
  def getNumberOfFeatures = alphabet.getSize
}

/**
 * Maps input lines in a sparse-vector format `<label> <f1>:<val> <f2>:<val> ... `
 * to a ''sparse'' vector input. This is most efficient if the inputs have high dimensionality
 * and are sparse. 
 * @param alphabet Feature alphabet
 * @param la Label alphabet
 * @author wellner
 */
class SparseVecFeatureExtractor(alphabet: Alphabet, la: Alphabet)
  extends FeatureExtractor[String, GLPFactor] with LineParser with Serializable {
  var ind = 0
  
  def getAlphabet = alphabet
  
  def extractFeatures(s: String): GLPFactor = {
    val (l, spv, id) = sparseOfLine(s, alphabet, addBias = false)
    val spVec : SparseVec = SparseVec(alphabet.getSize)    
    spv foreach { f =>
      if (f.fid >= 0) {
        val fv = alphabet.getValue(f.fid, f.value)
        spVec.update(f.fid, fv)
      }
    }
    val l_ind = la.ofString(l)
    val lv = DenseVec.zeros(la.getSize)
    lv.update(l_ind,1.0) // one-hot encoding
    ind += 1
    spVec.cacheArrays 
    new SparseGLPFactor(ind, spVec, lv, id)    
  }
  def getNumberOfFeatures = alphabet.getSize
}

/**
 * Extractor that constructs `DenseVec` dense vectors from an
 * input sparse representation where the feature indices for each feature have already been computed.
 * E.g. `<label> 1:1.0 9:1.0 10:0.95 ... `
 * This is most efficient and avoids using another symbol table if
 * the features have already been mapped to integers - e.g. with datasets in libSVM/libLINEAR format. 
 * @author
 */
class StdVectorExtractorWithAlphabet(la: Alphabet, nfs: Int) extends FeatureExtractor[String, GLPFactor] with Serializable {
  val reader = new SparseToDenseReader(' ', nfs)
  
  def getAlphabet = new IdentityAlphabet(nfs)
  
  def extractFeatures(s: String) : GLPFactor = {
    val (lab, features) = reader.getLabeledLine(s)
    val targetVec = DenseVec.zeros(la.getSize)
    targetVec.update(la.ofString(lab), 1.0) // set one-hot
    new StdGLPFactor(features, targetVec)
  }
  def getNumberOfFeatures = nfs
}


/**
 * Utilities for mapping GLP factors to Spark DataFrames
 * @author wellner
 */
trait DataFrames {
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

class GlpModel extends DataFrames {
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
    val optimizer: DistributedOptimizerEstimator[GLPFactor, GLPWeights] = GLPOptimizer.getDistributedOptimizer(sc, appSettings, network, ev)
    val fvs = mapDfToGLPFactors(sc, trdata, trdata.columns.length - 1, components.labelAlphabet.getSize)
    val (w, _) = optimizer.estimate(fvs, Some(appSettings.numEpochs))
    GLPModelSpec(w, components.evaluator, components.labelAlphabet, components.featureExtractor)
  }
  
}

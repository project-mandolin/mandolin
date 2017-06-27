package org.mitre.mandolin.mlp.spark

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import org.mitre.mandolin.mlp._
import org.mitre.mandolin.optimize.spark.{DistributedOnlineOptimizer, DistributedOptimizerEstimator}
import org.mitre.mandolin.util.{Alphabet, IdentityAlphabet}
import org.mitre.mandolin.util.spark.SparkIOAssistant

/**
  * Utilities for mapping MMLP factors to Spark DataFrames
  *
  * @author wellner
  */

trait MMLPDataFrames {

  import org.apache.spark.sql.types.{StructType, StructField, DoubleType}
  import org.apache.spark.mllib.linalg.Vectors
  import org.apache.spark.sql._

  private def toDouble(x: Array[Float]) = x map {
    _.toDouble
  }

  def mapMMLPFactorsToDf(sqlSc: SQLContext, fvs: RDD[MMLPFactor], dim: Int) = {
    val im = (0 to dim).toArray
    val rows = fvs map { f =>
      val label = f.getOneHot.toDouble // label as the one-hot output index
    val fv = f.getInput
      val rseq = (label, Vectors.dense(toDouble(fv.asArray)))
      org.apache.spark.sql.Row.fromTuple(rseq)
    }
    val schema = StructType(Seq(StructField("label", DoubleType, true), StructField("features", new org.apache.spark.mllib.linalg.VectorUDT, true)))
    val df = sqlSc.createDataFrame(rows, schema)
    df
  }

  def mapMMLPFactorsToDf(sqlSc: SQLContext, fvs: RDD[MMLPFactor], alphabet: Alphabet) = {
    val im1 = alphabet.getInverseMapping.toList.sortWith { case ((k1, v1), (k2, v2)) => k1 < k2 }
    val im = (0, "label") :: im1
    val ar = im.toArray
    val rows = fvs map { f =>
      val label = f.getOneHot.toDouble // label as the one-hot output index
    val fv = f.getInput
      val rseq = (label, Vectors.dense(toDouble(fv.asArray)))
      org.apache.spark.sql.Row.fromTuple(rseq)
    }
    val schema = StructType(Seq(StructField("label", DoubleType, true), StructField("features", new org.apache.spark.mllib.linalg.VectorUDT, true)))
    val df = sqlSc.createDataFrame(rows, schema)
    df
  }

  /**
    * Map a Spark DataFrame back into an `RDD` of `MMLPFactor` objects.  Assumes the input
    * DataFrame just has two columns (label, features)
    */
  case class LabPoint(lab: Double, point: org.apache.spark.mllib.linalg.Vector)

  def mapDfToMMLPFactors(sc: SparkContext, df: Dataset[org.apache.spark.sql.Row], fvDim: Int, labelDim: Int): RDD[MMLPFactor] = {
    throw new RuntimeException("Spark 2.0 invalidates dataframe to MMLP code")
  }
}


class MMLPModel extends MMLPDataFrames with Serializable {

  import org.apache.spark.sql.DataFrame
  import org.apache.spark.sql._
  import org.mitre.mandolin.optimize.Updater
  import scala.reflect.ClassTag

  def copy() = new MMLPModel

  def readAsDataFrame(sqlContext: SQLContext, sc: SparkContext, fPath: String, idim: Int, odim: Int) = {
    val lines = sc.textFile(fPath)
    val dp = new DistributedProcessor()
    val fe = new StdVectorExtractorWithAlphabet(new IdentityAlphabet(odim, false), new IdentityAlphabet(idim), idim)
    val fvs = lines map {
      fe.extractFeatures
    }
    mapMMLPFactorsToDf(sqlContext, fvs, idim)
  }

  def evaluate(mspec: MMLPModelSpec, tstdata: DataFrame): org.mitre.mandolin.predict.DiscreteConfusion = {
    val sc = tstdata.sqlContext.sparkContext
    val dp = new DistributedProcessor()
    val dim = mspec.wts.getInputDim
    val odim = mspec.wts.getOutputDim
    val predictor = new CategoricalMMLPPredictor(mspec.ann, true)
    val evalDecoder = new org.mitre.mandolin.predict.spark.EvalDecoder(mspec.fe, predictor)
    val wBc = sc.broadcast(mspec.wts)
    val tstVecs = mapDfToMMLPFactors(sc, tstdata, dim, odim)
    evalDecoder.evalUnits(tstVecs, wBc)
  }

  def estimate(trdata: DataFrame, modelSpec: IndexedSeq[LType]): MMLPModelSpec = {
    if (modelSpec.length < 3) estimate(trdata, modelSpec, AdaGradSpec(0.1))
    else estimate(trdata, modelSpec, RMSPropSpec(0.001))
  }

  def estimate(trdata: DataFrame, mSpec: IndexedSeq[LType], upSpec: UpdaterSpec,
               epochs: Int = 20, threads: Int = 8, partitions: Int = 0): MMLPModelSpec = {
    val dp = new DistributedProcessor()
    val idim = trdata.select("features").head().get(0).asInstanceOf[org.apache.spark.mllib.linalg.Vector].size
    val odim = trdata.select("label").rdd map { v => v.getDouble(0) } max // get the maximum value of the label column
    val modelSpec = ANNetwork.fullySpecifySpec(mSpec, idim, odim.toInt + 1)
    val components = dp.getComponentsDenseVecs(modelSpec)
    val glp = components.ann
    val layout = glp.generateZeroedLayout
    upSpec match {
      case AdaGradSpec(lr) => estimate(components, trdata, modelSpec, new MMLPAdaGradUpdater(layout, lr.toFloat), epochs, threads, partitions)
      case RMSPropSpec(lr) => estimate(components, trdata, modelSpec, new MMLPRMSPropUpdater(layout, lr.toFloat), epochs, threads, partitions)
      case SgdSpec(lr) => estimate(components, trdata, modelSpec, new MMLPSgdUpdater(layout, false, lr.toFloat), epochs, threads, partitions)
      case AdaDeltaSpec => estimate(components, trdata, modelSpec, new MMLPAdaDeltaUpdater(layout, layout.copy()), epochs, threads, partitions)
      case NesterovSgdSpec(lr) =>
        val np = trdata.count().toInt // get total number of training points to scale momentum in Nesterov accelerated SGD
        estimate(components, trdata, modelSpec, new MMLPSgdUpdater(layout, true, lr.toFloat, numPoints = np), epochs, threads, partitions)
    }
  }

  def estimate[U <: Updater[MMLPWeights, MMLPLossGradient, U] : ClassTag](components: MMLPComponentSet, trdata: DataFrame,
                                                                          modelSpec: IndexedSeq[LType], updater: U, epochs: Int, threads: Int, numPartitions: Int): MMLPModelSpec = {
    val sc = trdata.sqlContext.sparkContext
    val dim = components.dim
    val odim = components.labelAlphabet.getSize
    val glp = components.ann
    val ev = new MMLPInstanceEvaluator[U](glp)
    val optimizer =
      new DistributedOnlineOptimizer[MMLPFactor, MMLPWeights, MMLPLossGradient, U](sc, glp.generateRandomWeights, ev, updater,
        epochs, 1, threads, None)
    val fvs1 = mapDfToMMLPFactors(sc, trdata, dim, odim)
    val fvs = if (numPartitions > 0) fvs1.repartition(numPartitions) else fvs1
    fvs.persist()
    val (w, _) = optimizer.estimate(fvs)
    MMLPModelSpec(w, glp, components.labelAlphabet, components.featureExtractor)
  }

  def estimate(trdata: DataFrame, appSettings: MandolinMLPSettings): MMLPModelSpec = {
    val dp = new DistributedProcessor(appSettings.numPartitions)
    val io = new SparkIOAssistant(trdata.sqlContext.sparkContext)
    val components = dp.getComponentsViaSettings(appSettings, io)
    val fe = components.featureExtractor
    val trainFile = appSettings.trainFile
    val sc = AppConfig.getSparkContext(appSettings)
    val numPartitions = appSettings.numPartitions
    val network = components.ann
    val optimizer: DistributedOptimizerEstimator[MMLPFactor, MMLPWeights] = DistributedMMLPOptimizer.getDistributedOptimizer(sc, appSettings, network)
    val fvs1 = mapDfToMMLPFactors(sc, trdata, trdata.columns.length - 1, components.labelAlphabet.getSize)
    val fvs = if (numPartitions > 0) fvs1.repartition(numPartitions) else fvs1
    fvs.persist()
    val (w, _) = optimizer.estimate(fvs, Some(appSettings.numEpochs))
    MMLPModelSpec(w, network, components.labelAlphabet, components.featureExtractor)
  }

}

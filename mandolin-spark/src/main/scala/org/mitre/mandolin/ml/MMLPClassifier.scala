package org.mitre.mandolin.ml
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.mlp.{ StdMMLPFactor, MMLPModelSpec, LType, CategoricalMMLPPredictor,
  SoftMaxLType, InputLType, UpdaterSpec, AdaGradSpec}
import org.mitre.mandolin.mlp.spark.MMLPModel
import org.mitre.mandolin.util.{DenseTensor1 => DenseVec}
import org.apache.spark.ml.{Predictor, PredictionModel}
import org.apache.spark.ml.param._
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.linalg.{Vector => MllibVector}

/**
 * `spark.ml` parameters that control MMLP classifier settings
 * @author wellner
 */
trait MMLPParams extends Params {
  val modelSpecParam = new Param[IndexedSeq[LType]](this, "modelSpec", "Model specification")
  setDefault(modelSpecParam -> IndexedSeq(LType(InputLType), LType(SoftMaxLType)))

  val maxIterParam = new IntParam(this,"maxIter","Maximum number of iterations over entire dataset for training")
  setDefault(maxIterParam -> 20)

  val updaterSpecParam = new Param[UpdaterSpec](this, "updaterSpec", "Updater specification")
  setDefault(updaterSpecParam -> AdaGradSpec(0.1))

  val threadsParam = new IntParam(this,"numThreads","Number of threads to use on each worker node/partition")
  setDefault(threadsParam -> 8)

  val numPartitionsParam = new IntParam(this, "numPartitions", "Number of data partitions(nodes) for training")
  setDefault(numPartitionsParam -> 0)
}

/**
 * Classification model based on a general multi-layered perceptron. Includes fast online (asynchronous)
 * parameter estimation on each worker node with periodic (per epoch) model/weight averaging. Often much
 * faster than batch or mini-batch gradient descent with computation of gradient spread across the cluster.
 * MandolinMain implementation here includes a general way to specify multi-layer perceptrons with many different
 * possible activation functions, network topologies, loss functions and forms of regularization.
 * @param uid Unique id for this object (randomly generated if not provided)
 * @param mmlp A MMLP model instance
 * @author wellner
 */
abstract class MMLPClassifier(override val uid: String, mmlp: MMLPModel)
extends Predictor[MllibVector, MMLPClassifier, MMLPClassificationModel] with MMLPParams {

  def this(mmlp: MMLPModel) = this(Identifiable.randomUID("MMLPClassifier"), mmlp)

  def copy(extra: ParamMap) : MMLPClassifier // = new MMLPClassifier(mlp.copy())
  def setLayerSpec(sp: IndexedSeq[LType]) : this.type = set(modelSpecParam, sp)
  def setUpdaterSpec(usp: UpdaterSpec) : this.type = set(updaterSpecParam, usp)
  def setMaxIters(mi: Int) : this.type = set(maxIterParam, mi)
  def setThreads(th:Int) : this.type   = set(threadsParam, th)
  def setNumPartitions(p: Int) : this.type = set(numPartitionsParam, p)

  /**
   * Trains a MMLP model; other functions use this, notably `fit`.
   * @param dataset A training dataset
   * @return A `MMLPClassificationModel` used to make predictions
   */
  def train(dataset: DataFrame) : MMLPClassificationModel = {
    val modelSpec = get(modelSpecParam).get
    val updaterSpec = get(updaterSpecParam).getOrElse(AdaGradSpec(0.1))
    val maxIters = get(maxIterParam).getOrElse(20)
    val threads  = get(threadsParam).getOrElse(8)
    val partitions = get(numPartitionsParam).getOrElse(0)
    val m = mmlp.estimate(dataset, modelSpec, updaterSpec, maxIters, threads, partitions)
    new MMLPClassificationModel(m)
  }
}

/**
 * Classification model based on a general multi-layer perceptron
 * @param uid Unique id
 * @param gmSpec An object that specifies a fully trained MMLP model, including feature extraction and label/feature alphabets
 * @author wellner
 */
class MMLPClassificationModel(override val uid: String, val gmSpec: MMLPModelSpec) extends PredictionModel[MllibVector, MMLPClassificationModel] {
  def this(gmSpec: MMLPModelSpec) = this(Identifiable.randomUID("MMLPClassificationModel"), gmSpec)

  protected val numLabels = gmSpec.la.getSize
  private val predictor = new CategoricalMMLPPredictor(gmSpec.ann, false)

  def copy(extra: ParamMap) : MMLPClassificationModel = new MMLPClassificationModel(gmSpec.copy())
  
  def predict(input: MllibVector) : Double = {
    val ar = new DenseVec(input.toArray)
    val out = DenseVec.zeros(numLabels)
    val mmlpFactor = new StdMMLPFactor(-1, ar, out, None)
    predictor.getPrediction(mmlpFactor, gmSpec.wts).toDouble
  }
}

abstract class MMLPEvaluator(mmlp: MMLPModel, gmSpec: MMLPModelSpec) extends Evaluator {

  import org.apache.spark.sql.{Dataset, Row}
  
  def evaluateROC(df: Dataset[Row]) : Double = {
    val res = mmlp.evaluate(gmSpec, df)
    res.getTotalAreaUnderROC()
  }
}
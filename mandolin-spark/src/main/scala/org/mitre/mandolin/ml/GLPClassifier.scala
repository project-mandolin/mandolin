package org.mitre.mandolin.ml
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.glp.{GLPFactor, StdGLPFactor, GLPModelSpec, LType, GLPPredictor, 
  SoftMaxLType, InputLType, UpdaterSpec, AdaGradSpec}
import org.mitre.mandolin.glp.spark.GlpModel
import org.mitre.mandolin.util.{DenseTensor1 => DenseVec}
import org.apache.spark.ml.{Predictor, PredictorParams, PredictionModel}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector => MllibVector}

/**
 * `spark.ml` parameters that control GLP classifier settings
 * @author wellner
 */
trait GLPParams extends Params {
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
 * Mandolin implementation here includes a general way to specify multi-layer perceptrons with many different
 * possible activation functions, network topologies, loss functions and forms of regularization.
 * @param uid Unique id for this object (randomly generated if not provided)
 * @param glp A Glp model instance
 * @author wellner
 */
class GLPClassifier(override val uid: String, glp: GlpModel) 
extends Predictor[MllibVector, GLPClassifier, GLPClassificationModel] with GLPParams {
  
  def this(glp: GlpModel) = this(Identifiable.randomUID("GLPClassifier"), glp)
  
  def copy(extra: ParamMap) : GLPClassifier = new GLPClassifier(glp.copy())
  def setLayerSpec(sp: IndexedSeq[LType]) : this.type = set(modelSpecParam, sp)
  def setUpdaterSpec(usp: UpdaterSpec) : this.type = set(updaterSpecParam, usp)
  def setMaxIters(mi: Int) : this.type = set(maxIterParam, mi)
  def setThreads(th:Int) : this.type   = set(threadsParam, th)
  def setNumPartitions(p: Int) : this.type = set(numPartitionsParam, p)

  /**
   * Trains a GLP model; other functions use this, notably `fit`.
   * @param dataset A training dataset
   * @return A `GLPClassificationModel` used to make predictions
   */
  override def train(dataset: DataFrame) : GLPClassificationModel = {
    val modelSpec = get(modelSpecParam).get
    val updaterSpec = get(updaterSpecParam).getOrElse(AdaGradSpec(0.1))
    val maxIters = get(maxIterParam).getOrElse(20)
    val threads  = get(threadsParam).getOrElse(8)
    val partitions = get(numPartitionsParam).getOrElse(0)
    val m = glp.estimate(dataset, modelSpec, updaterSpec, maxIters, threads, partitions)
    new GLPClassificationModel(m)
  }  
}

/**
 * Classification model based on a general multi-layer perceptron
 * @param uid Unique id
 * @param gmSpec An object that specifies a fully trained GLP model, including feature extraction and label/feature alphabets
 * @author wellner
 */
class GLPClassificationModel(override val uid: String, val gmSpec: GLPModelSpec) extends PredictionModel[MllibVector, GLPClassificationModel] {
  def this(gmSpec: GLPModelSpec) = this(Identifiable.randomUID("GLPClassificationModel"), gmSpec)
  
  protected val numLabels = gmSpec.la.getSize
  private val predictor = new GLPPredictor(gmSpec.ann, false)
  
  def copy(extra: ParamMap) : GLPClassificationModel = new GLPClassificationModel(gmSpec.copy())
  
  def predict(input: MllibVector) : Double = {
    val ar = new DenseVec(input.toArray)
    val out = DenseVec.zeros(numLabels)
    val glpFactor = new StdGLPFactor(-1, ar, out, None)
    predictor.getPrediction(glpFactor, gmSpec.wts).toDouble
  }
}

abstract class GLPEvaluator(glp: GlpModel, gmSpec: GLPModelSpec) extends Evaluator {

  def evaluate(df: DataFrame) : Double = {
    val res = glp.evaluate(gmSpec, df)
    res.getTotalAreaUnderROC()
  }
}
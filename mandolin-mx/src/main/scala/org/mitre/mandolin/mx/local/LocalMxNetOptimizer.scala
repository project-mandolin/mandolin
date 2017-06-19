package org.mitre.mandolin.mx.local

import org.mitre.mandolin.optimize.local.LocalOnlineOptimizer
import org.mitre.mandolin.mx._
import org.mitre.mandolin.glp.GLPFactor
import ml.dmlc.mxnet.{Symbol, Context, Shape, NDArray, Uniform, Xavier}
import ml.dmlc.mxnet.optimizer.SGD

/**
  * This follows the LocalGLPOptimizer but really just thinly wraps MxNet
  * allowing for data conventions used by Mandolin and to ensure that local
  * and Spark-based usage is consistent.
  */
object LocalMxNetOptimizer {

  val batchSize = 64

  // hardcoded example
  def getMlp: Symbol = {
    val data = Symbol.Variable("data")
    val fc1 = Symbol.FullyConnected(name = "fc1")()(Map("data" -> data, "num_hidden" -> 1000))
    val act1 = Symbol.Activation(name = "relu1")()(Map("data" -> fc1, "act_type" -> "relu"))
    val fc2 = Symbol.FullyConnected(name = "fc2")()(Map("data" -> act1, "num_hidden" -> 1000))
    val act2 = Symbol.Activation(name = "relu2")()(Map("data" -> fc2, "act_type" -> "relu"))
    val fc3 = Symbol.FullyConnected(name = "fc3")()(Map("data" -> act2, "num_hidden" -> 10))
    val mlp = Symbol.SoftmaxOutput(name = "softmax")()(Map("data" -> fc3))
    mlp
  }

  val uniInit = new Uniform(0.1f)
  val xavierInit = new Xavier(factorType = "in", magnitude = 2.32f)

  def initializeParameters(net: Symbol, inputShapes: Map[String, Shape]) = {
    val (argShapes, _, auxShapes) = net.inferShape(inputShapes)
    val argNames = net.listArguments()
    val inputNames = inputShapes.keys
    val paramNames = argNames.toSet -- inputNames.toSet
    val auxNames = net.listAuxiliaryStates()

    val paramNameShapes = (argNames zip argShapes).filter { case (name, _) =>
      paramNames.contains(name)
    }
    val argParams = paramNameShapes.map { case (name, shape) =>
      (name, NDArray.zeros(shape))
    }.toMap
    val auxParams = (auxNames zip auxShapes).map { case (name, shape) =>
      (name, NDArray.zeros(shape))
    }.toMap
    argParams foreach { case (name, shape) => xavierInit(name, shape) }
    auxParams foreach { case (name, shape) => xavierInit(name, shape) }
    new MxNetWeights(argParams, auxParams, 1.0f)
  }

  def getLocalOptimizer() = {
    val sgd = new SGD(learningRate = 0.1f)
    val updater = new MxNetOptimizer(sgd)
    val inDim = 784
    val mlp = getMlp

    val evaluator = new MxNetGlpEvaluator(mlp, Array(Context.cpu(0), Context.cpu(1)), 784)
    val params = initializeParameters(mlp, Map("data" -> Shape(batchSize, 784), "softmax_label" -> Shape(batchSize)))
    val mxEpochs = 10
    val numSubEpochs = 1
    val workersPerPartition = 1
    val optDetails: Option[String] = None
    new LocalOnlineOptimizer[GLPFactor, MxNetWeights, MxNetLossGradient, MxNetOptimizer](params,
      evaluator, updater, mxEpochs, numSubEpochs, workersPerPartition, optDetails)
  }
}

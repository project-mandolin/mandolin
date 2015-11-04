package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.util.{ DenseTensor1 => DenseVec, DenseTensor2 => DenseMat, Tensor1 => Vec }

/**
 * Represents a feed-forward MLP represented as a sequence of Layer
 * @param inLayer the input layer
 * @param layers hidden layers and output layer
 * @param sparseInput true if inputs are sparse and a sparse vector representation should be used
 * @author wellner
 */
class ANNetwork(val inLayer: InputLayer, val layers: IndexedSeq[NonInputLayer], sparseInput: Boolean) extends Serializable {
  val revLayers = layers.reverse
  val outLayer = layers.last
  val numLayers = layers.length

  @inline
  final private def getGaussian(rv: util.Random, v: Double) = 
    rv.nextGaussian() * v

  private def getLayerWeights(curDim: Int, prevDim: Int, lt: LType, allZero: Boolean): (DenseMat, DenseVec) = {
    val w = DenseMat.zeros(curDim, prevDim)
    if (!allZero) {
      val gv = math.sqrt(2.0 / curDim) 
      val rv = new scala.util.Random
      var i = 0; while (i < curDim) {
        var j = 0; while (j < prevDim) {
          if (util.Random.nextBoolean()) w.update(i, j, getGaussian(rv, gv))
          j += 1
        }
        i += 1
      }
    }
    val b = DenseVec.zeros(curDim)
    lt.designate match {
      case TanHLType => b += (0.5)
      case _ =>
    }
    (w, b)
  }

  /**
   * Generate a new random set of weights with appropriate layout/dimensions for this particular network structure
   * @author wellner
   */
  def generateRandomWeights: GLPWeights = {
    val zeroLastLayer = numLayers == 1
    val wts = IndexedSeq.tabulate(numLayers) { i =>
      if (i == 0) getLayerWeights(layers(i).dim, inLayer.dim, layers(i).ltype, zeroLastLayer)
      else getLayerWeights(layers(i).dim, layers(i - 1).dim, layers(i).ltype, false)
    }
    new GLPWeights(new GLPLayout(wts))
  }

  /**
   * Generate an all zero layout of parameters for auxiliary functions such as Adagrad updaters, etc.
   * @author wellner
   */
  def generateZeroedLayout: GLPLayout = {
    val layout = IndexedSeq.tabulate(numLayers) { i =>
      val prevDim = if (i == 0) inLayer.dim else layers(i - 1).dim
      (DenseMat.zeros(layers(i).dim, prevDim), DenseVec.zeros(layers(i).dim))
    }
    new GLPLayout(layout)
  }

  /**
   * Perform a feed-forward pass with `inputVec` as inputs using the current
   * weights passed in as `glpW`
   */
  def forwardPass(inputVec: Vec, targetVec: DenseVec, glpW: GLPWeights, training: Boolean = true): Unit = {
    if ((numLayers > 1) || !sparseInput) {
      inLayer.setOutput(inputVec)
      outLayer.setTarget(targetVec)
      for (i <- 0 until numLayers) {
        val (w, b) = glpW.wts.get(i)
        layers(i).forward(w, b, training)
      }
    } else { // special-case with a single layer network (perceptron)
      val (w, b) = glpW.wts.get(0)
      outLayer.forwardWith(inputVec, w, b, training)
    }
  }

  /** Get the full gradient (over all layers) for the input `inputVec` provided the
   *  target output vector `targetVec`, using the model weights `glpW`
   *  @return GLPLayout object representing the gradient as a sequence of (matrix, vector) pairs 
   */
  def getGradient(inputVec: Vec, targetVec: DenseVec, glpW: GLPWeights) = {
    forwardPass(inputVec, targetVec, glpW, true)
    if ((numLayers > 1) || !sparseInput) {
      val revGrads = for (i <- (numLayers - 1) to 0 by -1) yield {
        val (w, b) = glpW.wts.get(i)
        layers(i).getGradient(w, b) // backward pass and return gradient at that layer
      }
      new GLPLayout(revGrads.reverse) // return in order from input to penultimate layer
    } else {
      val (w, b) = glpW.wts.get(0)
      new GLPLayout(IndexedSeq(outLayer.getGradientWith(inputVec, targetVec, w, b)))
    }
  }

  /** Get the cost or ''loss'' for this output derived from forward pass */
  def getCost = outLayer.getCost
  
  /** Get a deep copy of this network, including a new copy of the model weights */
  def copy() = {
    val inCopy = inLayer.copy()
    val nlayers = layers map { l: NonInputLayer => l.copy() }
    for (i <- 1 until layers.length) nlayers(i).setPrevLayer(nlayers(i - 1))
    nlayers(0).setPrevLayer(inCopy)
    new ANNetwork(inCopy, nlayers, sparseInput)
  }

  /** Get a shallow copy that has new layer structures, but shares the model weights */
  def sharedWeightCopy() = {
    val inCopy = inLayer.sharedWeightCopy()
    val nlayers = layers map { l: NonInputLayer => l.sharedWeightCopy() }
    for (i <- 1 until layers.length) nlayers(i).setPrevLayer(nlayers(i - 1))
    nlayers(0).setPrevLayer(inCopy)
    new ANNetwork(inCopy, nlayers, sparseInput)
  }
}

/**
 * Builds an ANN from a list of layer specification/types
 * @author wellner
 */
object ANNetwork {

  /** Construct network from list representation of layer specifications */
  def apply(specs: List[LType]): ANNetwork = apply(specs.toIndexedSeq)

  /**
   * Construct network from layer specifications
   * @param aspecs A sequence of layer specifications [[LType]]
   * @return An [[ANNetwork]] object representing an artificial neural network with specified topology
   * @author wellner
   */
  def apply(aspecs: IndexedSeq[LType]): ANNetwork = {
    val specs = aspecs.toIndexedSeq
    val lastInd = aspecs.length - 1
    val (inLayer, sp) = specs(0).designate match {
      case SparseInputLType => (new SparseInputLayer(specs(0).dim, specs(0).drO), true)
      case InputLType       => (new DenseInputLayer(specs(0).dim, specs(0).drO), false)
      case _                        => throw new RuntimeException("Invalid input layer specification: " + specs(0))
    }
    val olSp = sp && (lastInd < 2)
    val layers: IndexedSeq[NonInputLayer] =
      for (i <- 1 to lastInd) yield {
        var prevDim = specs(i - 1).dim
        val lt = specs(i)
        val d = lt.dim
        lt.designate match {
          case TanHLType   => WeightLayer.getTanHLayer(lt, prevDim, i)
          case LogisticLType     => WeightLayer.getLogisticLayer(lt, prevDim, (i == lastInd), i)
          case LinearLType       => WeightLayer.getLinearLayer(lt, prevDim, (i == lastInd), i)
          case CrossEntropyLType => WeightLayer.getCELayer(lt, prevDim, (i == lastInd), i)
          case ReluLType         => WeightLayer.getReluLayer(lt, prevDim, i)
          case SoftMaxLType           => WeightLayer.getOutputLayer(new SoftMaxLoss(d), lt, prevDim, i, olSp)
          case HingeLType(c)             => WeightLayer.getOutputLayer(new HingeLoss(d, c), lt, prevDim, i, olSp)
          case ModHuberLType          => WeightLayer.getOutputLayer(new ModifiedHuberLoss(d), lt, prevDim, i, olSp)
          case RampLType(rl)              => WeightLayer.getOutputLayer(new RampLoss(d,rl), lt, prevDim, i, olSp)
          case TransLogLType          => WeightLayer.getOutputLayer(new TransLogLoss(d), lt, prevDim, i, olSp)
          case TLogisticLType         => WeightLayer.getOutputLayer(new TLogisticLoss(d), lt, prevDim, i, olSp)
          case _                                  => throw new RuntimeException("Invalid non-input layer specification: " + specs(i))
        }
      }
    for (i <- 1 until layers.length) { layers(i).setPrevLayer(layers(i - 1)) }
    layers(0).setPrevLayer(inLayer)
    new ANNetwork(inLayer, layers.toIndexedSeq, sp)
  }
  
  /**
   * Construct a network from an underspecified layer spec.
   * Specifically, the input dimensions and output dimensions haven't been
   * provided in the layer spec and are provided here to properly instantiate the network
   * @param aspecs A sequence of layer specifications (with Input and Output dims not yet specified)
   * @param idim Number of input dimensions
   * @param odim Number of output dimensions (labels)
   * @return A new sequence of specs instantiated with input and output dimensions
   * @author wellner
   */
  def fullySpecifySpec(aspecs: IndexedSeq[LType], idim: Int, odim: Int) : IndexedSeq[LType] = {
    val lastId = aspecs.length - 1
    val nsp = for (i <- 0 until aspecs.length) yield {
      if (i == 0) aspecs(i).copy(dim = idim)
      else if (i == lastId) aspecs(i).copy(dim = odim)
      else aspecs(i)
    }
    nsp
  }
  
  /**
   * Construct a network from an underspecified layer spec.
   * Specifically, the input dimensions and output dimensions haven't been
   * provided in the layer spec and are provided here to properly instantiate the network
   * @param aspecs A sequence of layer specifications (with Input and Output dims not yet specified)
   * @param idim Number of input dimensions
   * @param odim Number of output dimensions (labels)
   * @return An [[ANNetwork]] object representing and artificial neural network with topology provided
   * @author wellner
   */
  def apply(aspecs: IndexedSeq[LType], idim: Int, odim: Int) : ANNetwork = {
    apply(fullySpecifySpec(aspecs, idim, odim))
  }
}

package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.util.{ StdAlphabet, RandomAlphabet, Alphabet }
import org.mitre.mandolin.util.{
  DenseTensor1 => DenseVec,
  DenseTensor2 => DenseMat,
  Tensor1 => Vec,
  SparseTensor1 => SparseVec,
  ColumnSparseTensor2 => SparseMat,
  Tensor2 => Mat
}

sealed abstract class LayerDesignate extends Serializable
case object InputLType extends LayerDesignate
case object SparseInputLType extends LayerDesignate
case class SparseSeqInputLType(vocabSize: Int) extends LayerDesignate
case object SeqEmbeddingLType extends LayerDesignate
case object TanHLType extends LayerDesignate
case object LogisticLType extends LayerDesignate
case object LinearLType extends LayerDesignate
case object CrossEntropyLType extends LayerDesignate
case object ReluLType extends LayerDesignate
case object SoftMaxLType extends LayerDesignate

/** @param c The coeficient defining the size of the margin*/
case class  HingeLType(c: Double = 1.0) extends LayerDesignate

case object ModHuberLType extends LayerDesignate

/** @param rl The length (on x-axis) of ramp */
case class RampLType(rl: Double = 2.0) extends LayerDesignate
case object TransLogLType extends LayerDesignate
case object TLogisticLType extends LayerDesignate

case class LType(val designate: LayerDesignate, 
    val dim: Int = 0, val drO: Double = 0.0, val l1: Double = 0.0, val l2: Double = 0.0, val maxNorm: Double = 0.0) 
extends Serializable

abstract class Layer(val index: Int, val dim: Int, val ltype: LType) extends Serializable {
  /* Reference to previous layer in the network */
  var prevLayer: Option[Layer] = None

  /* Gets the previous layer in the network */
  def prev: Layer = prevLayer.get

  // this stores the errors for backprop
  val delta: DenseVec = DenseVec.zeros(dim)

  def getOutput: Vec = getOutput(false)
  def getOutput(training: Boolean): Vec
  def setOutput(v: Vec) : Unit

}

/**
 * A single layer in an MLP. Stores weights and gradients (for weight layers).
 * @author wellner
 */
abstract class NonInputLayer(i: Int, val d: Int, lt: LType) extends Layer(i, d, lt) with Serializable {

  // this is the inputs to the layer after weights applied
  protected val output: DenseVec = DenseVec.zeros(dim)

  def setOutput(v: Vec) : Unit = v match {
    case vv: DenseVec => setOutput(vv)
    case _ => throw new RuntimeException("Non-Input layers must have dense outputs")
  }
  
  def setOutput(v: DenseVec) : Unit = {
    output := v
  }

  def setPrevLayer(l: Layer): Unit

  /*
   * Method to get output from this layer.  Input layers may have sparse 'outputs'
   */
  def getOutput(tr: Boolean): Vec = output

  def getTarget: DenseVec

  def setTarget(v: DenseVec): Unit

  /* Get the gradients as a pair: gradient matrix and bias vector */
  def getGradient: (Mat, DenseVec)
  
  def getGradient(w: Mat, b: DenseVec) : (Mat, DenseVec)
  
  def getGradientWith(in: Vec, out: DenseVec, w: Mat, b: DenseVec) : (Mat, DenseVec)

  def getActFnDeriv: DenseVec

  def getCost: Double

  /* 
   * Updates the parameter values by the gradients scaled input parameter
   * @param v scalar value applied to gradients before update 
   * */
  def updateScalar(v: Double): Unit = {}

  /* Feed-forward using inputs present in the input layer */
  def forward(w: Mat, b: DenseVec, training: Boolean): Unit
  
  def forwardWith(in: Vec, w:Mat, b: DenseVec, training: Boolean): Unit

  def copy(): NonInputLayer

  def sharedWeightCopy(): NonInputLayer

}

/*
 * Input layer simply holds a single input (one training instance)
 * @author wellner
 */
abstract class InputLayer(d: Int, lt: LType) extends Layer(0, d, lt) {

  def getOutput(tr: Boolean): Vec
  def copy(): InputLayer
  def sharedWeightCopy(): InputLayer
}

class DenseInputLayer(_d: Int, drO: Double = 0.0) extends InputLayer(_d, LType(InputLType, _d, drO)) {

  protected val output: DenseVec = DenseVec.zeros(dim)
  def setOutput(v: Vec): Unit = v match { case x: DenseVec => setOutput(x) case _ => throw new RuntimeException("Input vector type mis-match") }
  def setOutput(v: DenseVec): Unit = { output := v }
  def getOutput(tr: Boolean): Vec = {
    if (tr && (drO > 0.0)) { output.addMaskNoise(drO) }
    output
  }
  def copy() = new DenseInputLayer(_d, drO)
  def sharedWeightCopy() = new DenseInputLayer(_d, drO)

}

class SparseInputLayer(_d: Int, drO: Double = 0.0) extends InputLayer(_d, LType(SparseInputLType, _d, drO)) {
  protected val output: SparseVec = SparseVec(dim)
  def setOutput(v: Vec): Unit = v match { case x: SparseVec => setOutput(x) case _ => throw new RuntimeException("Input vector type mis-match") }
  def setOutput(v: SparseVec) = { output := v }
  def getOutput(tr: Boolean): Vec = output
  def copy() = new SparseInputLayer(_d, drO)
  def sharedWeightCopy() = new SparseInputLayer(_d)
}

/*
 * Abstract weight layer
 * @param curDim number of nodes in this layer
 * @param prevDim number of nodes in previous layer
 * @param outputLayer whether this is an output layer or not
 * @param i the index of this layer (0 - input, n-1 output)  
 * @author
 */
abstract class AbstractWeightLayer(val curDim: Int, val prevDim: Int, outputLayer: Boolean, i: Int,
                                   actFn: DenseVec => DenseVec,
                                   actFnDeriv: DenseVec => DenseVec,
                                   costFn: (DenseVec, DenseVec) => Double,
                                   costGradFn: (DenseVec, DenseVec) => DenseVec,
                                   lt: LType,
                                   dropOut: Double = 0.0) extends NonInputLayer(i, curDim, lt) with Serializable {

  def this(cd: Int, pd: Int, ol: Boolean, i: Int, loss: LossFunction, lt: LType, dOut: Double) =
    this(cd, pd, ol, i, loss.link _, loss.linkGrad _, loss.loss _, loss.lossGrad _, lt, dOut)

  def this(cd: Int, pd: Int, ol: Boolean, i: Int, loss: LossFunction, lt: LType) =
    this(cd, pd, ol, i, loss, lt, lt.drO)

  val mbuf = if (dropOut > 0.0) Array.fill[Boolean](curDim)(false) else Array.fill[Boolean](1)(false)
  // this is the target vector, generally just for output layers
  val target: Option[DenseVec] = if (outputLayer) Some(DenseVec.zeros(dim)) else None

  def setTarget(v: DenseVec) = { target.get := v }
  def getTarget = target.get

  def getActFnDeriv = actFnDeriv(output)
  def setPrevLayer(l: Layer) = { prevLayer_=(Some(l)) }
  
  def forwardWith(in: Vec, w: Mat, b: DenseVec, training: Boolean) = {
    val drO = dropOut
    if (!outputLayer) {
      if (drO > 0.0) {
        if (drO > 0.01) {
          if (training) {
            var i = 0; while (i < curDim) {
              if (util.Random.nextDouble < drO) mbuf(i) = false else mbuf(i) = true
              i += 1
            }
            w *= (in, output, mbuf) // matrix vector product with row mask
            output += (b, mbuf) // add vector to vector with same mask
            output := actFn(output) // XXX - this could be sped up if actiation function worked on components rather than vectors
          } else { // at prediction time, scale the outputs if we've used dropout
            w *= (in, output)
            output += b
            output := actFn(output)
            output *= (1.0 - drO)
          }
        } else throw new RuntimeException("Dropout value of " + drO + " not allowed")
      }
    }
    // standard forward prop if this is output layer or no dropout
    if (outputLayer || (drO < 0.01)) {
      w *= (in, output)
      output += b
      output := actFn(output)
    }
  }

  /*
   * Implements row dropout.  This will only work for activation functions
   * with the property that a(0) = 0 - e.g. ReLU or TanH
   */
  def forward(w: Mat, b: DenseVec, training: Boolean = true) = {  
    val prevIn = prev.getOutput(training)
    forwardWith(prevIn, w, b, training)    
  }
  
}

class WeightLayer(curDim: Int, _prevDim: Int, outputLayer: Boolean, i: Int,
                  actFn: DenseVec => DenseVec,
                  actFnDeriv: DenseVec => DenseVec,
                  costFn: (DenseVec, DenseVec) => Double,
                  costGradFn: (DenseVec, DenseVec) => DenseVec,
                  lt: LType,
                  dropOut: Double = 0.0)
  extends AbstractWeightLayer(curDim, _prevDim, outputLayer, i, actFn, actFnDeriv, costFn, costGradFn, lt, dropOut) {
  
  def this(cd: Int, pd: Int, ol: Boolean, i: Int, loss: LossFunction, lt: LType, dOut: Double) =
    this(cd, pd, ol, i, loss.link _, loss.linkGrad _, loss.loss _, loss.lossGrad _, lt, dOut)

  def this(cd: Int, pd: Int, ol: Boolean, i: Int, loss: LossFunction, lt: LType) =
    this(cd, pd, ol, i, loss, lt, lt.drO)
  

  val grad: Mat = DenseMat.zeros(curDim, prevDim)
  val bgrad: DenseVec = DenseVec.zeros(curDim)
  
  def getCost = costFn(getTarget, output)
  
  def getGradient(w: Mat, b: DenseVec) : (Mat, DenseVec) = {
    backward(w: Mat, b: DenseVec)
    getGradient
  }
  
  def getGradientWith(in: Vec, out: DenseVec, w: Mat, b: DenseVec) : (Mat, DenseVec) = 
    getGradient(w, b)
  
  def getGradient : (Mat, DenseVec) = (grad, bgrad)
  
  
  private def backward(w: Mat, b: DenseVec) = {
    val deriv: DenseVec = actFnDeriv(output)
    if (outputLayer) { // hardcoded here for CE-loss or losses with the same form (t - o)
      delta := getTarget
      delta -= output
      delta *= deriv
    }
    prev match {
      case p: NonInputLayer =>
        w.trMult(delta, p.delta)
        p.delta *= p.getActFnDeriv
      case _ =>
    }
    bgrad := delta
    grad.outerFill(delta, prev.getOutput(true))

  }

  def copy() = {
    val cur = this
    val nl = new WeightLayer(curDim, prevDim, outputLayer, i, actFn, actFnDeriv, costFn, costGradFn, lt, dropOut) {
      override val grad = cur.grad.copy
      override val bgrad = cur.bgrad.copy
    }
    nl
  }

  def sharedWeightCopy() = {
    val cur = this
    val nl = new WeightLayer(curDim, prevDim, outputLayer, i, actFn, actFnDeriv, costFn, costGradFn, lt, dropOut) {
      override val grad = cur.grad.copy
      override val bgrad = cur.bgrad.copy
    }
    nl
  }
}

class SparseGradientWeightLayer(curDim: Int, prevDim: Int, outputLayer: Boolean, i: Int,
                                actFn: DenseVec => DenseVec,
                                actFnDeriv: DenseVec => DenseVec,
                                costFn: (DenseVec, DenseVec) => Double,
                                costGradFn: (DenseVec, DenseVec) => DenseVec,
                                lt: LType,
                                dropOut: Double = 0.0) 
                                extends AbstractWeightLayer(curDim, prevDim, outputLayer, i, actFn, actFnDeriv, costFn, costGradFn, lt, dropOut) {

  def this(cd: Int, pd: Int, ol: Boolean, i: Int, loss: LossFunction, lt: LType, dOut: Double) =
    this(cd, pd, ol, i, loss.link _, loss.linkGrad _, loss.loss _, loss.lossGrad _, lt, dOut)

  def this(cd: Int, pd: Int, ol: Boolean, i: Int, loss: LossFunction, lt: LType) =
    this(cd, pd, ol, i, loss, lt, lt.drO)
    
    
  private var lastCost = 0.0
  
  def getCost : Double = lastCost 

  def copy() = {
    val cur = this
    val nl = new SparseGradientWeightLayer(curDim, prevDim, outputLayer, i, actFn, actFnDeriv, costFn, costGradFn, lt, dropOut)
    nl
  }

  def sharedWeightCopy() = {
    val cur = this
    val nl = new SparseGradientWeightLayer(curDim, prevDim, outputLayer, i, actFn, actFnDeriv, costFn, costGradFn, lt, dropOut)
    nl
  }
  
  def getGradient(w: Mat, b: DenseVec) = getGradientWith(prev.getOutput(true), output, w, b)
  
  def getGradientWith(in: Vec, target: DenseVec, w: Mat, b: DenseVec) = {
    val deriv: DenseVec = actFnDeriv(output)
    val d = target.copy
    d -= output
    d *= deriv
    val grad : Mat = SparseMat.zeros(curDim, prevDim)
    grad.outerFill(d, in)
    lastCost = costFn(target, output)
    (grad, d)
  }
  
  def getGradient: (Mat, DenseVec) = throw new RuntimeException("Unimplemented")

}

object WeightLayer {
  private def getSECost(t: DenseVec, o: DenseVec): Double = {
    var n = 0.0
    var i = 0; while (i < t.getDim) {
      val d = (t(i) - o(i))
      n += (d * d)
      i += 1
    }
    n * n / 2.0
  }

  private def getCECost(t: DenseVec, o: DenseVec): Double = {
    var r = 0.0
    var i = 0; while (i < t.getDim) {
      val fO = if (o(i) < 1E-10) 1E-10 else o(i)
      r += fO * t(i)
      i += 1
    }
    r
  }

  private final def getStraightCostGrad(t: DenseVec, o: DenseVec): DenseVec = t :- o

  def getTanHLayer(lt: LType, pd: Int, ind: Int) = {
    val thFn = { v: DenseVec => v.map { x => math.tanh(x) } }
    val thDeriv = { v: DenseVec => v.map { x => val th = math.tanh(x); 1.0 - th * th } }
    new WeightLayer(lt.dim, pd, false, ind, thFn, thDeriv, { (_, _) => 0.0 }, getStraightCostGrad _, lt, lt.drO)
  }

  def getReluLayer(lt: LType, pd: Int, ind: Int) = {
    val thFn = { v: DenseVec => v map { x => if (x > 0.0) x else 0.0 } }
    val thDeriv = { v: DenseVec => v map { x => if (x > 0.0) 1.0 else 0.0 } }
    new WeightLayer(lt.dim, pd, false, ind, thFn, thDeriv, getSECost _, getStraightCostGrad _, lt, lt.drO)
  }

  def getLinearLayer(lt: LType, pd: Int, isOut: Boolean, ind: Int) = {
    val thFn = { v: DenseVec => v }
    val thDeriv = { v: DenseVec => DenseVec.ones(lt.dim) }
    new WeightLayer(lt.dim, pd, isOut, ind, thFn, thDeriv, getSECost _, getStraightCostGrad _, lt, lt.drO)
  }

  def getLogisticLayer(lt: LType, pd: Int, isOut: Boolean, ind: Int) = {
    val ones = DenseVec.ones(lt.dim)
    val thFn = { v: DenseVec => v map { v => 1.0 / (1.0 + math.exp(-v)) } }
    val thDeriv = { v: DenseVec => v :* (ones :- v) }
    new WeightLayer(lt.dim, pd, isOut, ind, thFn, thDeriv, getSECost _, getStraightCostGrad _, lt, lt.drO)
  }

  def getCELayer(lt: LType, pd: Int, isOut: Boolean, ind: Int) = {
    val ones = DenseVec.ones(lt.dim)
    val thFn = { v: DenseVec => v map { v => 1.0 / (1.0 + math.exp(-v)) } }
    val thDeriv = { v: DenseVec => ones }
    new WeightLayer(lt.dim, pd, isOut, ind, thFn, thDeriv, getCECost _, getStraightCostGrad _, lt, lt.drO)
  }

  def getOutputLayer(loss: LossFunction, lt: LType, pd: Int, ind: Int, sp: Boolean) : AbstractWeightLayer =
    if (sp)
      new SparseGradientWeightLayer(lt.dim, pd, true, ind, loss, lt, 0.0)
    else new WeightLayer(lt.dim, pd, true, ind, loss, lt, 0.0)
  
}


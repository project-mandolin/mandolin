package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.util.{DenseTensor1 => DenseVec, Tensor1 => Vec, SparseTensor1 => SparseVec}

/**
 * The loss function here captures both the output activation/link function but
 * also captures the loss of the activated/linked output vs. the target and its gradient.
 * This is a more flexible interface as it allows for regression losses, for example.
 * The "link" function here is the activation function applied to an <i>individual</i>
 * output node. In most cases this is linear. The full loss function takes the component outputs
 * after application of the link function to compute the loss. For example, 
 * a standard softmax layer uses an identity link function (each output neuron has a
 * linear activation) while the loss function applies the softmax function to these
 * outputs to get a final loss
 * @param d Dimension size of the output space 
 * @author wellner
 */
abstract class LossFunction(dim: Int) extends Serializable {
  @inline
  final def getMax(scores: DenseVec, y: Int) : (Float,Int) = {
    var mx = -Float.MaxValue    
    var maxI = 0
    var i = 0; while (i < dim) {
      val cs = scores(i)
      if ((i != y) && (cs > mx)) {
        mx = cs
        maxI = i
      }
      i += 1
    }
    (mx, maxI)
  }

  /**
   * Applies softmax to an input dense vector; scales values to avoid overflow
   * @return Unit normalized dense vector 
   */
  final def softMax(v: DenseVec) : DenseVec = {
    val mx = v.maxVal
    var s = 0.0f
    val na = new DenseVec(Array.tabulate(v.getDim){i => 
      val n = math.exp(v(i) - mx).toFloat
      s += n
      n
      })
    na *= (1.0f / s)
    na
  }
  
  /**
   * Value of the loss comparing target output vector to predicted output
   * @return real-valued loss
   */
  def loss(t: DenseVec, o: DenseVec) : Double
  
  /**
   * Gradient of the loss function
   * @return gradient vector
   */
  def lossGrad(t: DenseVec, o: DenseVec) : DenseVec
  
  /**
   * Link/activation function. This is a function applied to each output
   * component - e.g. a squashing function. Identity function for linear link.
   * @return outputs after application of link/activation function
   */
  def link(o: DenseVec) : DenseVec
  
  /**
   * Computes the gradient of the link/activation function. A vector
   * of ones for linear activations on the output layer.
   */
  def linkGrad(o: DenseVec) : DenseVec
}

/**
 * Implements a multiclass Hinge Loss (cf. Crammer and Singer, 2004) 
 * @param d Dimension size of the output space
 * @author wellner
 */
class HingeLoss(d: Int, val coef: Double = 1.0) extends LossFunction(d) {
  val ones = DenseVec.ones(d)
  def linkGrad(o: DenseVec) = ones
  def link(o: DenseVec) = o
  def loss(t: DenseVec, o: DenseVec) = {
    val argMxT = t.argmax // assume target is a one-hot vector 
    val scMxT  = o(argMxT)
    val (mx,_) = getMax(o, argMxT)
    math.max(0.0, coef + (mx - scMxT))
  }
  
  def lossGrad(t: DenseVec, o: DenseVec) = {
    val y = t.argmax
    val trueScore = o(y)
    val (mx,mxI) = getMax(o, y) 
    val dim = o.getDim
    val gr = DenseVec.zeros(dim)
    if ((mx - trueScore) > -coef) { // if outside margin coef1
      gr(mxI) = -1.0f
      gr(y) = 1.0f
    }
    gr
  }  
}

/**
 * Standard softmax layer with standard cross-entropy loss
 * @param d Dimension size of the output space
 * @author wellner
 */
class SoftMaxLoss(d: Int) extends LossFunction(d) {
  val ones = DenseVec.ones(d)
  def linkGrad(o: DenseVec) = ones  
  
  def link(v: DenseVec) = { softMax(v) }
  
  def loss(t: DenseVec, o: DenseVec) = {
    var r = 0.0
    var i = 0
    val mlog = math.log(1E-10)
    while (i < t.getDim) {
      val fO = if (o(i) < 1E-10) mlog else math.log(o(i))
      r += fO * t(i)
      i += 1
    }
    r
  }  
  def lossGrad(t: DenseVec, o: DenseVec) = t :- o  
}

/**
 * Squared error loss with linear activations
 * @author wellner
 */
class SquaredErrorLoss(d: Int) extends LossFunction(d) {
  val ones = DenseVec.ones(d)
  def linkGrad(o: DenseVec) = ones  
  def link(v: DenseVec) = v
  def loss(t: DenseVec, o: DenseVec) = {
    var n = 0.0
    var i = 0; while (i < t.getDim) {
      val c = (t(i) - o(i))
      n += c * c
      i += 1
    }
    n / 2.0
  }  
  def lossGrad(t: DenseVec, o: DenseVec) = {
    t :- o
  }  
}

/**
 * A modified Huber loss for classification; adapted for multiclass
 * @param d Dimension size of the output space
 * @author wellner
 */
class ModifiedHuberLoss(d: Int) extends LossFunction(d) {
  val ones = DenseVec.ones(d)
  def linkGrad(o: DenseVec) = ones  
  
  def link(v: DenseVec) = v
  def loss(t: DenseVec, scores: DenseVec) = {
    val y = t.argmax
    val trueScore = scores(y) // score for true category
    val (mx, _) = getMax(scores, y)
    val nsc = mx - trueScore
    if (nsc < -1.0) 0.0
    else if (nsc < 1.0) ((1.0 + nsc) * (1.0 + nsc))
    else nsc * 4.0
  }
  
  def lossGrad(t: DenseVec, scores: DenseVec) = {
    val y = t.argmax 
    val trueScore = scores(y)
    val (mx, mxI) = getMax(scores, y)
    var i = 0; while (i < scores.getDim) {
      scores.update(i, 0.0f)
      i += 1
    }
    val nsc = mx - trueScore
    val d =
      if (nsc > -1.0f) {
        if (nsc < 1.0f) 2.0f * nsc
        else 4.0f
      } else 0.0f
    scores(mxI) -= d
    scores(y) += d
    scores
  }
}

/**
 * A ramp loss function assigns a linear penalty outside the margin (1.0)
 * but assigns NO penalty to predictions with a larger margin(2.0), thus
 * the function has a "ramp" shape. This non-convex loss can provided better
 * results in the face of "label noise" as it tends to ignore some 
 * mis-classifications.
 * @param d Dimension size of the output space
 * @author wellner
 */
class RampLoss(d: Int, val beta: Double = 2.0) extends LossFunction(d) {
  val ones = DenseVec.ones(d)
  
  def linkGrad(o: DenseVec) = ones  
  
  def link(v: DenseVec) = v
  def loss(t: DenseVec, o: DenseVec) = {
    val argMxT = t.argmax // assume target is a one-hot vector 
    val scMxT  = o(argMxT)  // model score for true output
    val (mx,_) = getMax(o, argMxT)
    math.max(0.0, 1.0 + math.min(beta, (mx - scMxT)))
  }
  
  def lossGrad(t: DenseVec, o: DenseVec) = {
    val y = t.argmax
    val trueScore = o(y)
    val (mx,mxI) = getMax(o, y) 
    val dim = o.getDim
    val gr = DenseVec.zeros(dim)
    val diff = mx - trueScore 
    if ((diff > -1.0) && (diff < beta)) { // if outside margin coef1
      gr(mxI) = -1.0f
      gr(y) = 1.0f
    }
    gr
  }
}

/**
 * Special case of T-logistic loss; with multiclass extensions
 * @param d Dimension size of the output space
 * @author wellner
 */
class TransLogLoss(d: Int) extends LossFunction(d) {
  val ones = DenseVec.ones(d)

  val q : Float = 1.0f // controls shape of arcsignh; closer to linear as q -> 0.0 
  def linkGrad(o: DenseVec) = ones
  
  private final def arcSinH(v: Float) = {
    val qv = q*v
    math.log(qv + math.sqrt(1.0 + qv*qv)).toFloat / q
  }
  
  private final def derivArcSinH(v: Float) = {
    val qv = q*v
    1.0f / math.sqrt(1.0f + qv * qv).toFloat
  }
  
  private final def getWeights(scores: DenseVec) = {
    new DenseVec(Array.tabulate(scores.getDim){i => -derivArcSinH(scores(i)) })
  }
  
  private final def transSoftMax(scores: DenseVec) = {
    // transform scores
    val ln = scores.getDim
    var i = 0; while(i < ln) {
      scores(i) = arcSinH(scores(i))
      i += 1
    }
    softMax(scores)
  }
  
  def link(v: DenseVec) = v
  def loss(t: DenseVec, o: DenseVec) = {
    val y = t.argmax
    val ts = transSoftMax(o)
    -(math.log(ts(y)))
  }
  
  def lossGrad(t: DenseVec, o: DenseVec) = {
    val y = t.argmax
    val gr = o.copy
    val weights = getWeights(gr)
    transSoftMax(gr) 
    gr *= weights    
    gr(y) -= weights(y) 
    gr
  }
  
}

/**
 * A smooth non-convex family of loss functions. See the paper
 * "t-Logistic Regression", N. Ding and S.V.N. Vishwanathan, 2011. NIPS
 * This is a multiclass extension of the loss.
 * @param d Dimension size of the output space
 * @author wellner
 */
class TLogisticLoss(d: Int) extends LossFunction(d) {
  val t : Float = 1.0f
  
  val eps: Float = 0.1f
  val mxEpochs = 10
  val ones = DenseVec.ones(d)
  
  private def tExp(v: Float) : Float = {
    if (t == 1) math.exp(v).toFloat else {
      val base = 1.0f + (1.0f - t) * v
      if (base <= 0.0) 0.0f
      else math.pow(base,(1.0f / (1.0f - t))).toFloat
    }
  }
  
  private def logt(v: Float) : Float = {
    if (t == 1) math.log(v).toFloat
    else {
      math.pow(v,(1.0f - t) / (1.0f - t)).toFloat
    }
  }
  
  private final def getPartition(scores: Array[Float]) : Float = {
    var mx = -Float.MaxValue
    val ln = scores.length
    var i = 0; while (i < ln) {
      if (scores(i) > mx) mx = scores(i)
      i += 1
    }
    i = 0; while (i < ln) {
      scores(i) -= mx
      i += 1
    }
    val uhat = Array.tabulate(ln)(i => scores(i))
    val uhatTmp = Array.tabulate(ln)(i => scores(i))
    var converged = false
    var l2Norm = 0.0
    var epochs = 1
    var z = 0.0f
    while (!converged) {
      z = 0.0f
      i = 0; while (i < ln) {
        z += tExp(uhat(i))
        i += 1
      }
      val zp = math.pow(z,(1.0 - t)).toFloat
      l2Norm = 0.0
      i = 0; while (i < ln) {
        uhatTmp(i) = zp * scores(i)
        val diff = (uhat(i) - uhatTmp(i)) 
        l2Norm += diff * diff
        i += 1
      }
      if ((l2Norm < eps) || epochs > mxEpochs) converged = true
      else { // not converged, set uhat to uhatTmp
        i = 0; while (i < ln) {
          uhat(i) = uhatTmp(i)
          i += 1
        }
      }
      epochs += 1
    }
    mx - logt(1.0f / z)
  }
  
  private final def escortDistrib(dist: Array[Float], t: Float) = {
    var s = 0.0f
    val ln = dist.length
    var i = 0; while (i < ln) {
      dist(i) = math.pow(dist(i), t).toFloat
      s += dist(i)
      i += 1
    }
    i = 0; while (i < ln) {
      dist(i) /= s
      i += 1
    }
  }
  
  private final def getWeights(scores: DenseVec, t: Float) = {
    new DenseVec(Array.tabulate(scores.getDim){i => -math.pow(scores(i),(t - 1.0f)).toFloat})
  }
  
  private final def tSoftMax(scores: DenseVec) : DenseVec = {
    val l = scores.getDim
    val origScores = Array.tabulate(l){i => scores(i)}
    val res = scores.copy
    val zp = getPartition(origScores)
    var i = 0; while (i < l) { // this will help handle large dot products that could result in overflow
      res(i) = tExp(scores(i) - zp)
      i += 1
    }
    res
  }
    
  def linkGrad(o: DenseVec) = ones
  def link(v: DenseVec) = v
  def loss(tgt: DenseVec, o: DenseVec) = {
    val y = tgt.argmax
    val ts = tSoftMax(o)
    -(math.log(ts(y)))
  }
  
  def lossGrad(tgt: DenseVec, o: DenseVec) = {
    val y = tgt.argmax
    val gr = o.copy
    val weights = getWeights(o, t)
    escortDistrib(gr.a, t)    
    gr *= weights    
    gr(y) -= weights(y) 
    gr
  }  
}

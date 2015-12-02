package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.optimize.{ Weights, Updater, LossGradient }
import org.mitre.mandolin.predict.{ EvalPredictor, DiscreteConfusion }
import org.mitre.mandolin.util.{ DenseTensor2 => DenseMat, ColumnSparseTensor2 => SparseMat, DenseTensor1 => DenseVec, Tensor2 => Mat, Tensor1 => Vec }

abstract class ComposeStrategy
case object Minimum extends ComposeStrategy
case object Maximum extends ComposeStrategy
case object Average extends ComposeStrategy

/**
 * Holds the data structures used to represent weights and gradients for GLP
 * models. These are simply an `IndexedSeq` with elements consisting of
 * pairs of type (`Mat`,`DenseVec`), each pair representing the weights and bias
 * for a single layer of the network
 * @author wellner
 */
class GLPLayout(val w: IndexedSeq[(Mat, Vec)]) extends Serializable {
  val length = w.length
  val totalSize = {
    var n = 0
    w foreach { case (w, b) => n += w.getSize; n += b.getDim }
    n
  }

  def getInputDim = w(0)._1.getDim2 // number of columns corresponds to number of inputs

  def getOutputDim = w.last._2.getDim

  @inline
  final def get(i: Int) = w(i)

  def checkValues(): Unit = {
    for (i <- 0 until length) {
      val (thisW, thisB) = this.get(i)
      //thisW.numericalCheck()
      //thisB.numericalCheck()
    }
  }

  def addEquals(other: GLPLayout, sc: Double = 1.0): Unit = {
    val o: GLPLayout = other
    assert(this.length == other.length)
    for (i <- 0 until length) {
      other.get(i) match {
        case (w, b) =>
          val (thisW, thisB) = this.get(i)
          if (sc != 1.0) {
            w *= sc
            thisW += w
            b *= sc
            thisB += b
          } else {
            thisW += w
            thisB += b
          }
      }
    }
  }

  def addEquals(v: Double): Unit = {
    for (i <- 0 until length) {
      get(i) match {
        case (w, b) =>
          w += v
          b += v
      }
    }
  }

  
  def set(v: Double): Unit = {
    for (i <- 0 until length) {
      get(i) match {
        case (w, b) =>
          w := v
          b := v
      }
    }
  }
  
  def timesEquals(v: Double): Unit = {
    for (i <- 0 until length) {
      get(i) match {
        case (w, b) =>
          w *= v
          b *= v
      }
    }
  }

  def copy(): GLPLayout = {
    new GLPLayout(w map { case (w, b) => (w.copy, b.copy) })
  }

  /*
   * This copies the layout structure but assigns all matrices, vectors a zero value
   */
  def copyZero(): GLPLayout = {
    new GLPLayout(w map {
      _ match {
        case (w, b) =>
          val nw: DenseMat = DenseMat.zeros(w.getDim1, w.getDim2)
          val nb: DenseVec = DenseVec.zeros(b.getDim)
          (nw, nb)
      }
    })
  }

  def print(): Unit = {
    for (i <- 0 until length) {
      get(i) match {
        case (w, b) =>
          println("LAYER: " + i)
          println("------------")
          println(w)
          println(b)
      }
    }
  }
}

/**
 * GLPLayout objects are constructed using this factory object
 * @author wellner
 */
object GLPLayout {
  def apply(w: IndexedSeq[(Mat, DenseVec)]) = new GLPLayout(w)
}

/**
 * GLP weights implements Mandolin Weight class, just wraps GLPLayout
 * @author wellner
 */
class GLPWeights(val wts: GLPLayout, m: Double) extends Weights[GLPWeights](m) with Serializable {
  def this(wts: GLPLayout) = this(wts, 1.0)

  val nlayers = wts.length
  def compress(): Unit = {}
  def decompress(): Unit = {}
  def weightAt(i: Int) = throw new RuntimeException("Not implemented")

  override def checkWeights() = {
    wts.checkValues()
  }

  def getInputDim = wts.getInputDim
  def getOutputDim = wts.getOutputDim

  val numWeights: Int = wts.totalSize

  def compose(otherWeights: GLPWeights) = {
    this *= mass
    otherWeights *= otherWeights.mass
    this ++ otherWeights
    val nmass = mass + otherWeights.mass
    this *= (1.0 / nmass)
    new GLPWeights(this.wts, nmass)
  }

  def add(otherWeights: GLPWeights): GLPWeights = {
    this += otherWeights
    this
  }

  def addEquals(otherWeights: GLPWeights): Unit = wts addEquals otherWeights.wts

  def timesEquals(v: Double) = { wts.timesEquals(v) }

  def l2norm = throw new RuntimeException("Norm not implemented yet")

  def updateFromArray(ar: Array[Double]) = {
    if (wts.length > 1) throw new RuntimeException("array update for NN not implemented")
    else {
      val (m,b) = wts.get(0)      
      System.arraycopy(ar, 0, m.asArray, 0, m.getSize)
      System.arraycopy(ar, m.getSize, b.asArray, 0, b.getSize)
      this
    }
  }

  def copy() = new GLPWeights(wts.copy(), m)

  def asArray() = {
    if (wts.length > 1) throw new RuntimeException("Array form not available ")
    else {
      val (m,b) = wts.get(0)
      val ma = m.asArray
      val ar = Array.fill(m.getSize + b.getSize)(0.0)
      System.arraycopy(m.asArray, 0, ar, 0, m.getSize)
      System.arraycopy(b.asArray, 0, ar, m.getSize, b.getSize)
      ar
    }
  }
  def asTensor1() = throw new RuntimeException("Tensor construction not available with deep weight array")

  override def toString(): String = {
    val sb = new StringBuilder
    for (i <- 0 until wts.length) {
      val (w, b) = wts.get(i)
      sb append ("Weights: \n")
      sb append w.toString()
      sb append ("Bias: \n")
      sb append b.toString()
      sb append "\n\n"
    }
    sb.toString
  }
}

class BasicGLPSgdUpdater(val initialLearningRate: Double = 0.2, lambda: Double = 0.1) extends Updater[GLPWeights, GLPLossGradient, BasicGLPSgdUpdater] {
  var numIterations = 0
  def copy() = {
    val sgd = new BasicGLPSgdUpdater(initialLearningRate)
    sgd.numIterations = this.numIterations
    sgd
  }
  def resetLearningRates(v: Double) = {}
  def compose(u: BasicGLPSgdUpdater) = this
  def updateWeights(lossGrad: GLPLossGradient, weights: GLPWeights): Unit = {
    val eta_t = initialLearningRate / (1.0 + (initialLearningRate * numIterations * lambda))
    weights.wts.addEquals(lossGrad.gr, eta_t)
    numIterations += 1
  }
}

class GLPLossGradient(l: Double, val gr: GLPLayout) extends LossGradient[GLPLossGradient](l) {
  def add(other: GLPLossGradient) = {
    gr.addEquals(other.gr)
    new GLPLossGradient(l + other.loss, gr)
  }
  def asArray =
    if (gr.length == 1) {
      val pa = gr.get(0)._1.asArray
      val bias = gr.get(0)._2.asArray
      val ar = Array.fill(pa.length + bias.length)(0.0)
      System.arraycopy(pa, 0, ar, 0, pa.length)
      System.arraycopy(bias, 0, ar, pa.length, bias.length)
      ar
    } else throw new RuntimeException("As Array not feasible with non-linear model")
}

trait Regularizer {
  @inline
  final private def fastSqrt(x: Double) =
    java.lang.Double.longBitsToDouble(((java.lang.Double.doubleToLongBits(x) >> 32) + 1072632448) << 31)

   /** implementation of scaling - cf. Hinton et al "Improving neural networks by preventing co-adaptation of feature detectors" */
  def rescaleWeightsDense(w_w: DenseMat, w_b: Vec, d1: Int, d2: Int, maxLayerSumSq: Double) = {              
    var i = 0; while (i < d1) {
      var ssq = 0.0
      var j = 0; while (j < d2) {
        ssq += w_w(i, j) * w_w(i, j)
        j += 1
      }
      ssq += (w_b(i) * w_b(i))
      if (ssq > maxLayerSumSq) {
        val sc = fastSqrt(maxLayerSumSq / ssq)
        j = 0; while (j < d2) {
          w_w.update(i, j, w_w(i, j) * sc)
          j += 1
        }
        w_b.update(i, w_b(i) * sc)
      }
      i += 1
    }
  }
  
     /** implementation of scaling - cf. Hinton et al "Improving neural networks by preventing co-adaptation of feature detectors" */
  def rescaleWeightsSparse(w_w: SparseMat, w_b: Vec, d1: Int, maxLayerSumSq: Double) = {              
    if (maxLayerSumSq > 0.0) {
    var i = 0; while (i < d1) {
      var ssq = 0.0
      val row = w_w(i)
      row.forEach((i,v) => ssq += v * v)
      ssq += (w_b(i) * w_b(i))
      if (ssq > maxLayerSumSq) {
        val sc = fastSqrt(maxLayerSumSq / ssq)
        row *= sc
        w_b.update(i, w_b(i) * sc)
      }
      i += 1
    }
    }
  }
}

sealed abstract class UpdaterSpec
case class AdaGradSpec(learnRate: Double = 0.1) extends UpdaterSpec
case class SgdSpec(learnRate: Double = 0.1) extends UpdaterSpec
case class NesterovSgdSpec(learnRate: Double = 0.1) extends UpdaterSpec
case class RMSPropSpec(learnRate: Double = 0.001) extends UpdaterSpec
case object AdaDeltaSpec extends UpdaterSpec

class GLPAdamUpdater(val alpha: Double, beta1: Double, beta2: Double, val mom1: GLPLayout, val mom2: GLPLayout, 
    composeSt: ComposeStrategy = Minimum, 
    maxNormArray: Option[Array[Double]] = None, l1Array: Option[Array[Double]] = None, l2Array: Option[Array[Double]] = None)
  extends Updater[GLPWeights, GLPLossGradient, GLPAdamUpdater] with Regularizer {

  val beta1Inv = 1.0 - beta1
  val beta2Inv = 1.0 - beta2
  val epsilon = 1E-8
  
  var numIterations = 0
  var beta1T = beta1
  var beta2T = beta2

  val nLayers = mom1.length
  def copy() = {
    val sgd = new GLPAdamUpdater(alpha, beta1, beta2, mom1, mom2, composeSt, maxNormArray, l1Array, l2Array)
    sgd.numIterations = this.numIterations
    sgd.beta1T = this.beta1T
    sgd.beta2T = this.beta2T
    sgd
  }
  def resetLearningRates(v: Double) = {}
  
  def compose(u: GLPAdamUpdater) = {
    for (i <- 0 until nLayers) {      
      mom1.get(i) match {
        case (th_w, th_b) =>
          val (ot_w, ot_b) = u.mom1.get(i)
          composeSt match {
            case Maximum =>
              th_w.mapInto(ot_w, { math.max })
              th_b.mapInto(ot_b, { math.max })
            case _ =>
              th_w.mapInto(ot_w, { math.min })
              th_b.mapInto(ot_b, { math.min })
          }
      }
      mom2.get(i) match {
        case (th_w, th_b) =>
          val (ot_w, ot_b) = u.mom2.get(i)
          composeSt match {
            case Maximum =>
              th_w.mapInto(ot_w, { math.max })
              th_b.mapInto(ot_b, { math.max })
            case _ =>
              th_w.mapInto(ot_w, { math.min })
              th_b.mapInto(ot_b, { math.min })
          }
      }
    }
    this
  }

  @inline
  final private def fastSqrt(x: Double) =
    java.lang.Double.longBitsToDouble(((java.lang.Double.doubleToLongBits(x) >> 32) + 1072632448) << 31)

  def updateWeights(lossGrad: GLPLossGradient, weights: GLPWeights): Unit = {
    beta1T *= beta1
    beta2T *= beta2
    for (l <- 0 until nLayers) {           
      val l1 = l1Array match {case Some(ar) => ar(l) case None => 0.0}
      val l2 = l2Array match {case Some(ar) => ar(l) case None => 0.0}
      val maxNorm = maxNormArray match {case Some(ar) => ar(l) case None => 0.0}
      val l1Reg = (l1 > 0.0)
      val l2Reg = (l2 > 0.0)

      (mom1.get(l), mom2.get(l)) match {
        case ((mom1_w, mom1_b), (mom2_w, mom2_b)) =>
          val (u_w, u_b) = lossGrad.gr.get(l)
          val (w_w, w_b) = weights.wts.get(l)
          val d1 = w_w.getDim1
          val d2 = w_w.getDim2
          var i = 0; while (i < d1) {
            u_w match {
              case u_w: DenseMat =>
                var j = 0; while (j < d2) {
                  val ww = w_w(i,j)
                  val cgr = if (l1Reg) u_w(i, j) - math.signum(ww) * l1 else if (l2Reg) u_w(i,j) - ww * l2 else u_w(i,j)
                  val mt = mom1_w(i, j) * beta1 + (cgr * beta1Inv)
                  val mt_hat = mt / (1.0 - beta1T)
                  mom1_w.update(i, j, mt)
                  val vt = mom2_w(i, j) * beta2 + (cgr * cgr * beta2Inv)
                  val vt_hat = vt / (1.0 - beta2T)
                  mom2_w.update(i, j, vt) 
                  //println("delta = " + (alpha * mt_hat / (fastSqrt(vt_hat) + epsilon)))
                  w_w.update(i, j, ww + (alpha * mt_hat / (fastSqrt(vt_hat) + alpha)))
                  j += 1
                }
              case u_w: SparseMat =>
                val row = u_w(i)
                row.forEach { (j, cgr_o) =>
                  val ww = w_w(i,j)
                  val cgr = if (l1Reg) cgr_o - math.signum(ww) * l1 else if (l2Reg) cgr_o - ww * l2 else cgr_o 
                  val mt = mom1_w(i, j) * beta1 + (cgr * beta1Inv)
                  val mt_hat = mt / (1.0 - beta1T)
                  mom1_w.update(i, j, mt)
                  val vt = mom2_w(i, j) * beta2 + (cgr * cgr * beta2Inv)
                  val vt_hat = vt / (1.0 - beta2T)
                  mom2_w.update(i, j, vt)                  
                  w_w.update(i, j, ww + (alpha * mt_hat / (fastSqrt(vt_hat) + alpha)))
                }                
            }
            val cgr = if (l1Reg) u_b(i) - math.signum(w_b(i)) * l1 else if (l2Reg) u_b(i) - w_b(i) * l2 else u_b(i)
            val mt = mom1_b(i) * beta1 + (cgr * beta1Inv)
            val mt_hat = mt / (1.0 - beta1T)
            mom1_b.update(i, mt)
            val vt = mom2_b(i) * beta2 + (cgr * cgr * beta2Inv)
            val vt_hat = vt / (1.0 - beta2T)
            mom2_b.update(i, vt)  
            //println("bias delta = " + (alpha * mt_hat / (fastSqrt(vt_hat) + epsilon)))
            w_b.update(i, w_b(i) + (alpha * mt_hat / (fastSqrt(vt_hat) + alpha)))
            i += 1
          }
          if (maxNorm > 0.0)
            w_w match { 
              case ww: DenseMat => rescaleWeightsDense(ww, w_b, d1, d2, maxNorm) 
              case ww: SparseMat => rescaleWeightsSparse(ww, w_b, d1, maxNorm)}
      }
    }
  }
}

/*
 * Updater for SGD with momentum, including Nesterov accelerated SGD (the default)
 * @param momentum layer of momentum values
 * @param nesterov use Nesterov acceleration - true by default
 * @param initialLearningRate non-decaying learning rate
 * @param maxLayerSumSq maximum some of squared weights as inputs to each neuron
 * @param numPoints total number of data points to scale/adjust momentum coefficient
 * @author wellner
 */
class GLPSgdUpdater(val momentum: GLPLayout, val nesterov: Boolean = true,
                    val initialLearningRate: Double = 0.05,
                    maxNormArray: Option[Array[Double]] = None, 
                    l1Array: Option[Array[Double]] = None, 
                    l2Array: Option[Array[Double]] = None, numPoints: Double = 500.0, compose: ComposeStrategy = Minimum)
  extends Updater[GLPWeights, GLPLossGradient, GLPSgdUpdater] with Regularizer {

  var numIterations = 0
  val nLayers = momentum.length
  
  def copy() = {
    val sgd = new GLPSgdUpdater(momentum, nesterov, initialLearningRate, maxNormArray, l1Array, l2Array, numPoints, compose)
    sgd.numIterations = this.numIterations
    sgd
  }
  def resetLearningRates(v: Double) = {}

  def compose(u: GLPSgdUpdater) = {
    for (i <- 0 until nLayers) {
      momentum.get(i) match {
        case (th_w, th_b) =>
          val (ot_w, ot_b) = u.momentum.get(i)
          compose match {
            case Maximum =>
              th_w.mapInto(ot_w, { math.max })
              th_b.mapInto(ot_b, { math.max })
            case _ =>
              th_w.mapInto(ot_w, { math.min })
              th_b.mapInto(ot_b, { math.min })
          }
      }
    }
    this
  }

  def updateWeights(lossGrad: GLPLossGradient, weights: GLPWeights): Unit = {
    val eta_t = initialLearningRate // / (1.0 + (initialLearningRate * numIterations * lambda))    
    numIterations += 1
    val momentumRate = 1.0 - (3.0 / (5.0 + (numIterations.toDouble / numPoints))) // cf. Sutsekver et al. for additional momentum schedules such as:
    // val momentumRate = math.min(0.9, (1.0 - math.pow(2.0,(-1.0 - math.log(math.floor(numIterations * 250 + 1))))))
    for (l <- 0 until nLayers) {
      val l1 = l1Array match {case Some(ar) => ar(l) case None => 0.0}
      val l2 = l1Array match {case Some(ar) => ar(l) case None => 0.0}
      val maxNorm = maxNormArray match {case Some(ar) => ar(l) case None => 0.0}
      val l1Reg = (l1 > 0.0)
      val l2Reg = (l2 > 0.0)
      
      
      momentum.get(l) match {
        case (mom_w, mom_b) =>
          val (u_w, u_b) = lossGrad.gr.get(l)
          val (w_w, w_b) = weights.wts.get(l)
          val d1 = w_w.getDim1
          val d2 = w_w.getDim2
          var i = 0; while (i < d1) {
            u_w match {
              case u_w: DenseMat =>
                var j = 0; while (j < d2) {
                  val ww = w_w(i,j)
                  val tv = if (l1Reg) u_w(i, j) - math.signum(ww) * l1 else if (l2Reg) u_w(i,j) - ww * l2 else u_w(i,j)
                    
                  if (nesterov) {
                    val nv = (momentumRate * mom_w(i, j)) + tv
                    mom_w.update(i, j, nv)
                    w_w.update(i, j, ww + nv * eta_t)
                  } else {
                    val tv_a = tv * eta_t
                    w_w.update(i, j, ww + tv_a + momentumRate * mom_w(i, j))
                    mom_w.update(i, j, tv_a)
                  }
                  j += 1
                }
              case u_w: SparseMat =>
                val row = u_w(i)
                row.forEach { (j, tvp) =>
                  val ww = w_w(i,j)
                  val tv = if (l1Reg) tvp - math.signum(ww) * l1 else if (l2Reg) tvp - ww * l2 else tvp
                  if (nesterov) {
                    val nv = (momentumRate * mom_w(i, j)) + tv
                    mom_w.update(i, j, nv)
                    w_w.update(i, j, ww + nv * eta_t)
                  } else {
                    val tv_a = tv * eta_t
                    w_w.update(i, j, ww + tv_a + momentumRate * mom_w(i, j))
                    mom_w.update(i, j, tv_a)
                  }
                }
            }
            val tv = if (l1Reg) u_b(i) - math.signum(w_b(i)) * l1 else if (l2Reg) u_b(i) - w_b(i) * l2 else u_b(i)              
            if (nesterov) {
              val nv = (momentumRate * mom_b(i)) + tv
              mom_b.update(i, nv)
              w_b.update(i, w_b(i) + nv * eta_t)
            } else {
              val tv_a = tv * eta_t
              w_b.update(i, w_b(i) + tv_a + momentumRate * mom_b(i))
              mom_b.update(i, tv_a)
            }
            i += 1
          }
          if (maxNorm > 0)
            w_w match { 
              case ww: DenseMat => rescaleWeightsDense(ww, w_b, d1, d2, maxNorm) 
              case ww: SparseMat => rescaleWeightsSparse(ww, w_b, d1, maxNorm)}
      }
    }
  }
}

class GLPRMSPropUpdater(val sumSquared: GLPLayout, val initialLearningRate: Double = 0.9, lambda: Double = 0.001,
                        rho: Double = 0.95, epsilon: Double = 0.003, maxNormArray: Option[Array[Double]] = None,
                        l1Array: Option[Array[Double]] = None, l2Array: Option[Array[Double]] = None, 
                        compose: ComposeStrategy = Minimum)
  extends Updater[GLPWeights, GLPLossGradient, GLPRMSPropUpdater] with Regularizer {

  sumSquared set initialLearningRate // set sum squared to initial learning rate

  val nLayers = sumSquared.length
  var numIterations = 0
  val rhoInv = 1.0 - rho
  
  /*
   * A <i>shallow</i> copy so learning rates are shared across threads/partitions on same machine
   */
  def copy() = new GLPRMSPropUpdater(sumSquared, initialLearningRate)

  def resetLearningRates(v: Double) = sumSquared.timesEquals(initialLearningRate * v)

  def compose(u: GLPRMSPropUpdater) = {
    for (i <- 0 until nLayers) {
      sumSquared.get(i) match {
        case (th_w, th_b) =>
          val (ot_w, ot_b) = u.sumSquared.get(i)
          compose match {
            case Maximum =>
              th_w.mapInto(ot_w, { math.max })
              th_b.mapInto(ot_b, { math.max })
            case _ =>
              th_w.mapInto(ot_w, { math.min })
              th_b.mapInto(ot_b, { math.min })
          }

      }
    }
    this
  }

  @inline
  final private def fastSqrt(x: Double) =
    java.lang.Double.longBitsToDouble(((java.lang.Double.doubleToLongBits(x) >> 32) + 1072632448) << 31)

  def updateWeights(lossGrad: GLPLossGradient, weights: GLPWeights): Unit = {
    // won't decay the learning rate since this is controled for via RMS update
    val eta_t = initialLearningRate // / (1.0 + (initialLearningRate * numIterations * lambda))
    numIterations += 1
    for (l <- 0 until nLayers) {
      val l1 = l1Array match {case Some(ar) => ar(l) case None => 0.0}
      val l2 = l2Array match {case Some(ar) => ar(l) case None => 0.0}
      val maxNorm = maxNormArray match {case Some(ar) => ar(l) case None => 0.0}
      val l1Reg = (l1 > 0.0)
      val l2Reg = (l2 > 0.0)
  
      sumSquared.get(l) match {
        case (sq_w, sq_b) =>
          val (u_w, u_b) = lossGrad.gr.get(l)
          val (w_w, w_b) = weights.wts.get(l)
          // updater mass gives an additional scalar weight/avg to gradient
          // this is used for mini-batch training
          if (updaterMass != 1.0) {
            u_w *= updaterMass
            u_b *= updaterMass
          }
          val d1 = w_w.getDim1
          val d2 = w_w.getDim2

          var i = 0; while (i < d1) {
            u_w match {
              case u_w: DenseMat =>
                var j = 0; while (j < d2) {
                  val ww = w_w(i,j)
                  val cgr = if (l1Reg) u_w(i, j) - math.signum(ww) * l1 else if (l2Reg) u_w(i,j) - ww * l2 else u_w(i,j)
                  val ss = sq_w(i, j) * rho + (cgr * cgr * rhoInv)
                  sq_w.update(i, j, ss)
                  val rms = fastSqrt(ss + epsilon)
                  w_w.update(i, j, ww + (eta_t * cgr / rms))
                  j += 1
                }
              case u_w: SparseMat =>
                val row = u_w(i)                 
                row.forEach { (j, cgr_o) =>
                  val ww = w_w(i,j)
                  val cgr = if (l1Reg) cgr_o - math.signum(ww) * l1 else if (l2Reg) cgr_o - ww * l2 else cgr_o
                  val ss = sq_w(i, j) * rho + (cgr * cgr * rhoInv)
                  sq_w.update(i, j, ss)
                  w_w.update(i, j, ww + (eta_t * cgr / fastSqrt(ss + epsilon)))
                }
            }
            val cgr = if (l1Reg) u_b(i) - math.signum(w_b(i)) * l1 else if (l2Reg) u_b(i) - w_b(i) * l2 else u_b(i)
            val ss = sq_b(i) * rho + (cgr * cgr * rhoInv)
            sq_b.update(i, ss)
            val rms = fastSqrt(ss + epsilon)
            w_b.update(i, w_b(i) + (eta_t * cgr / rms))
            i += 1
          }
          if (maxNorm > 0)
            w_w match { 
              case ww: DenseMat => rescaleWeightsDense(ww, w_b, d1, d2, maxNorm) 
              case ww: SparseMat => rescaleWeightsSparse(ww, w_b, d1, maxNorm) }
      }
    }
  }
}

class GLPAdaGradUpdater(val sumSquared: GLPLayout, val initialLearningRate: Double = 0.9, maxNormArray: Option[Array[Double]] = None,
    l1Array: Option[Array[Double]] = None, l2Array: Option[Array[Double]] = None,
    compose: ComposeStrategy = Minimum)
  extends Updater[GLPWeights, GLPLossGradient, GLPAdaGradUpdater] with Regularizer {

  sumSquared set initialLearningRate // set sum squared to initial learning rate

  val nLayers = sumSquared.length

  /*
   * A <i>shallow</i> copy so learning rates are shared across threads/partitions
   */
  def copy() = new GLPAdaGradUpdater(sumSquared.copy(), initialLearningRate, maxNormArray, l1Array, l2Array, compose)

  def resetLearningRates(v: Double) = sumSquared.timesEquals(initialLearningRate * v)

  def compose(u: GLPAdaGradUpdater) = {
    for (i <- 0 until nLayers) {
      sumSquared.get(i) match {
        case (th_w, th_b) =>
          val (ot_w, ot_b) = u.sumSquared.get(i)
          compose match {
            case Maximum =>
              th_w.mapInto(ot_w, { math.max })
              th_b.mapInto(ot_b, { math.max })
            case _ =>
              th_w.mapInto(ot_w, { math.min })
              th_b.mapInto(ot_b, { math.min })
          }

      }
    }
    this
  }

  @inline
  final private def fastSqrt(x: Double) =
    java.lang.Double.longBitsToDouble(((java.lang.Double.doubleToLongBits(x) >> 32) + 1072632448) << 31)

  def updateWeights(lossGrad: GLPLossGradient, weights: GLPWeights): Unit = {
    for (l <- 0 until nLayers) {
      val l1 = l1Array match {case Some(ar) => ar(l) case None => 0.0}
      val l2 = l2Array match {case Some(ar) => ar(l) case None => 0.0}
      val maxNorm = maxNormArray match {case Some(ar) => ar(l) case None => 0.0}
      val l1Reg = (l1 > 0.0)
      val l2Reg = (l2 > 0.0)
  
      sumSquared.get(l) match {
        case (sq_w, sq_b) =>
          val (u_w, u_b) = lossGrad.gr.get(l)
          val (w_w, w_b) = weights.wts.get(l)
          // updater mass gives an additional scalar weight/avg to gradient
          // this is used for mini-batch training
          if (updaterMass != 1.0) {
            u_w *= updaterMass
            u_b *= updaterMass
          }
          val d1 = w_w.getDim1
          val d2 = w_w.getDim2
          var i = 0; while (i < d1) {
            u_w match {
              case u_w: DenseMat =>
                var j = 0; while (j < d2) {
                  val ww = w_w(i, j)
                  val cgr = if (l1Reg) u_w(i, j) - math.signum(ww) * l1 else if (l2Reg) u_w(i,j) - ww * l2 else u_w(i,j) 
                  val csq = sq_w(i, j)
                  val nsq = csq + (cgr * cgr)
                  sq_w.update(i, j, nsq)                  
                  w_w.update(i, j, ww + (initialLearningRate * cgr / (initialLearningRate + fastSqrt(nsq))))
                  j += 1
                }
              case u_w: SparseMat =>
                val row = u_w(i)
                val inds = row.indArray
                val vls = row.valArray
                val ln = inds.length
                var o = 0; while (o < ln) {
                  val j = inds(o)
                  val ww = w_w(i, j)
                  val cgr_o = vls(o)
                  val cgr = if (l1Reg) cgr_o - math.signum(ww) * l1 else if (l2Reg) cgr_o - ww * l2 else cgr_o
                  val csq = sq_w(i, j)
                  val nsq = csq + (cgr * cgr)
                  sq_w.update(i, j, nsq)                  
                  w_w.update(i, j, ww + (initialLearningRate * cgr / (initialLearningRate + fastSqrt(nsq))))
                  o += 1
                }
            }
            val csq = sq_b(i)
            val cgr = if (l1Reg) u_b(i) - math.signum(w_b(i)) * l1 else if (l2Reg) u_b(i) - w_b(i) * l2 else u_b(i)
            val nsq = csq + (cgr * cgr)
            sq_b.update(i, nsq)
            val cw = w_b(i)
            w_b.update(i, cw + (initialLearningRate * cgr / (initialLearningRate + fastSqrt(nsq))))
            i += 1
          }
          if (maxNorm > 0.0) w_w match { 
            case ww: DenseMat => rescaleWeightsDense(ww, w_b, d1, d2, maxNorm) 
            case ww: SparseMat => rescaleWeightsSparse(ww, w_b, d1, maxNorm) }
      }
    }
  }
}

class GLPAdaDeltaUpdater(val sumSquared: GLPLayout, val prevUpdates: GLPLayout, val epsilon: Double = 0.001, val rho: Double = 0.95,
                         maxLayerSumSq: Double = 10.0, compose: ComposeStrategy = Minimum, maxNorm: Boolean = true)
  extends Updater[GLPWeights, GLPLossGradient, GLPAdaDeltaUpdater] with Regularizer {

  sumSquared set epsilon // set sum squared to epsilon

  val nLayers = sumSquared.length
  val rhoInv = 1.0 - rho

  @inline
  final private def fastSqrt(x: Double) =
    java.lang.Double.longBitsToDouble(((java.lang.Double.doubleToLongBits(x) >> 32) + 1072632448) << 31)

  /*
   * A <i>shallow</i> copy so learning rates are shared across threads/partitions
   */
  def copy() = new GLPAdaDeltaUpdater(sumSquared, prevUpdates, epsilon, rho)

  def resetLearningRates(v: Double) = throw new RuntimeException("not clear how to reset adadelta")

  def compose(u: GLPAdaDeltaUpdater) = {
    for (i <- 0 until nLayers) {
      sumSquared.get(i) match {
        case (th_w, th_b) =>
          val (ot_w, ot_b) = u.sumSquared.get(i)
          compose match {
            case Maximum =>
              th_w.mapInto(ot_w, { math.max })
              th_b.mapInto(ot_b, { math.max })
            case _ =>
              th_w.mapInto(ot_w, { math.min })
              th_b.mapInto(ot_b, { math.min })
          }
      }
      prevUpdates.get(i) match {
        case (pr_w, pr_b) =>
          val (opr_w, opr_b) = u.prevUpdates.get(i)
          pr_w.mapInto(opr_w, { (v1, v2) => math.max(v1, v2) })
          pr_b.mapInto(opr_b, { (v1, v2) => math.max(v1, v2) })
      }
    }
    this
  }

  def updateWeights(lossGrad: GLPLossGradient, weights: GLPWeights): Unit = {
    for (l <- 0 until nLayers) {
      sumSquared.get(l) match {
        case (sq_w, sq_b) =>
          val (prev_w, prev_b) = prevUpdates.get(l)
          val (u_w, u_b) = lossGrad.gr.get(l)
          val (w_w, w_b) = weights.wts.get(l)

          val d1 = w_w.getDim1
          val d2 = w_w.getDim2

          var i = 0; while (i < d1) {
            var j = 0; while (j < d2) {
              val cgr = u_w(i, j)
              val ss = sq_w(i, j) * rho + (cgr * cgr * rhoInv)
              sq_w.update(i, j, ss)
              val pr_w = prev_w(i, j)
              val update = cgr * fastSqrt((pr_w + epsilon) / (ss + epsilon))
              prev_w.update(i, j, (pr_w * rho + (update * update * rhoInv)))
              val cur = w_w(i, j)
              w_w.update(i, j, cur + update)
              j += 1
            }
            val cgr = u_b(i)
            val ss = sq_b(i) * rho + (cgr * cgr * rhoInv)
            sq_b.update(i, ss)
            val pr_b = prev_b(i)
            val update = cgr * fastSqrt((pr_b + epsilon) / (ss + epsilon))
            prev_b.update(i, (pr_b * rho + (update * update * rhoInv)))
            val cur = w_b(i)
            w_b.update(i, cur + update)
            i += 1
          }
          if (maxNorm) w_w match { case ww: DenseMat => rescaleWeightsDense(ww, w_b, d1, d2, maxLayerSumSq) case _ => }
      }
    }
  }
}

class GLPPredictor(network: ANNetwork, getScores: Boolean = false)
  extends EvalPredictor[GLPFactor, GLPWeights, Int, DiscreteConfusion] with Serializable {

  def getPrediction(u: GLPFactor, w: GLPWeights): Int = {
    network.forwardPass(u.getInput, u.getOutput, w, false)
    val uscores = network.outLayer.getOutput(false)
    uscores.argmax
  }

  def getScoredPredictions(u: GLPFactor, w: GLPWeights): Seq[(Double, Int)] = {
    network.forwardPass(u.getInput, u.getOutput, w, false)
    val uscores = network.outLayer.getOutput(false) // always softmax these scores
    uscores.asArray.toSeq.zipWithIndex
  }

  def getLoss(u: GLPFactor, w: GLPWeights): Double = {
    network.forwardPass(u.getInput, u.getOutput, w, false)
    network.getCost
  }

  def getConfusion(u: GLPFactor, w: GLPWeights): DiscreteConfusion = {
    val p = getPrediction(u, w)
    val ncats = network.outLayer.getOutput(false).getDim
    val targetVal = u.getOutput.argmax // output target is output with highest score (usually one-hot)
    if (getScores) {
      val scores = network.outLayer.getOutput(false).copy.asArray
      DiscreteConfusion(scores.length, p, targetVal, scores)
    } else DiscreteConfusion(ncats, p, targetVal)
  }
}



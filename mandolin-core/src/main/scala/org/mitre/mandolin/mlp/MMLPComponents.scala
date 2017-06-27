package org.mitre.mandolin.mlp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.optimize.{ Weights, Updater, LossGradient }
import org.mitre.mandolin.predict.{ EvalPredictor, DiscreteConfusion, Confusion, RegressionConfusion }
import org.mitre.mandolin.util.{ DenseTensor2 => DenseMat, ColumnSparseTensor2 => SparseMat, RowSparseTensor2 => RowSparseMat, 
  DenseTensor1 => DenseVec, Tensor2 => Mat, Tensor1 => Vec, SparseTensor1 => SparseVec }

abstract class ComposeStrategy
case object Minimum extends ComposeStrategy
case object Maximum extends ComposeStrategy
case object Average extends ComposeStrategy

/**
 * Holds the data structures used to represent weights and gradients for MMLP
 * models. These are simply an `IndexedSeq` with elements consisting of
 * pairs of type (`Mat`,`DenseVec`), each pair representing the weights and bias
 * for a single layer of the network
 * @author wellner
 */
class MMLPLayout(val w: IndexedSeq[(Mat, Vec)]) extends Serializable {
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

  def addEquals(other: MMLPLayout, sc: Float = 1.0f): Unit = {
    val o: MMLPLayout = other
    assert(this.length == other.length)
    for (i <- 0 until length) {
      other.get(i) match {
        case (w, b) =>
          val (thisW, thisB) = this.get(i)
          if (sc != 1.0f) {
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

  def addEquals(v: Float): Unit = {
    for (i <- 0 until length) {
      get(i) match {
        case (w, b) =>
          w += v
          b += v
      }
    }
  }


  def set(v: Float): Unit = {
    for (i <- 0 until length) {
      get(i) match {
        case (w, b) =>
          w := v
          b := v
      }
    }
  }

  def timesEquals(v: Float): Unit = {
    for (i <- 0 until length) {
      get(i) match {
        case (w, b) =>
          w *= v
          b *= v
      }
    }
  }

  def copy(): MMLPLayout = {
    new MMLPLayout(w map { case (w, b) => (w.copy, b.copy) })
  }

  /*
   * This copies the layout structure but assigns all matrices, vectors a zero value
   */
  def copyZero(): MMLPLayout = {
    new MMLPLayout(w map {
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
 * MMLPLayout objects are constructed using this factory object
 * @author wellner
 */
object MMLPLayout {
  def apply(w: IndexedSeq[(Mat, DenseVec)]) = new MMLPLayout(w)
}

/**
 * MMLP weights implements Mandolin Weight class, just wraps MMLPLayout
 * @author wellner
 */
class MMLPWeights(val wts: MMLPLayout, m: Float) extends Weights[MMLPWeights](m) with Serializable {
  def this(wts: MMLPLayout) = this(wts, 1.0f)

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

  def compose(otherWeights: MMLPWeights) = {
    this *= mass
    otherWeights *= otherWeights.mass
    this ++ otherWeights
    val nmass = mass + otherWeights.mass
    this *= (1.0f / nmass)
    new MMLPWeights(this.wts, nmass)
  }

  def add(otherWeights: MMLPWeights): MMLPWeights = {
    this += otherWeights
    this
  }

  def addEquals(otherWeights: MMLPWeights): Unit = wts addEquals otherWeights.wts

  def timesEquals(v: Float) = { wts.timesEquals(v) }

  def l2norm = throw new RuntimeException("Norm not implemented yet")

  def updateFromArray(ar: Array[Float]) = {
    if (wts.length > 1) throw new RuntimeException("array update for NN not implemented")
    else {
      val (m,b) = wts.get(0)
      System.arraycopy(ar, 0, m.asArray, 0, m.getSize)
      System.arraycopy(ar, m.getSize, b.asArray, 0, b.getSize)
      this
    }
  }

  def updateFromArray(ar: Array[Double]) = {
    throw new RuntimeException("array update for NN not implemented")
  }

  def copy() = new MMLPWeights(wts.copy(), m)

  def asArray() = {
    if (wts.length > 1) throw new RuntimeException("Array form not available ")
    else {
      val (m,b) = wts.get(0)
      val ma = m.asArray
      val ar = Array.fill(m.getSize + b.getSize)(0.0f)
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

class NullMMLPUpdater() extends Updater[MMLPWeights, MMLPLossGradient, NullMMLPUpdater] {

  def copy() = new NullMMLPUpdater
  def resetLearningRates(v: Float) = {}
  def compose(u: NullMMLPUpdater) = this
  def updateWeights(lossGrad: MMLPLossGradient, weights: MMLPWeights): Unit = {}
  def asArray : Array[Float] = throw new RuntimeException("As array not available for complex updater")
  def updateFromArray(ar: Array[Float]) = throw new RuntimeException("From array not available for complex updater")
  def compress() = this
  def decompress() = this

}

class BasicMMLPSgdUpdater(val initialLearningRate: Double = 0.2, lambda: Double = 0.1) extends Updater[MMLPWeights, MMLPLossGradient, BasicMMLPSgdUpdater] {
  var numIterations = 0

  val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)

  def asArray : Array[Float] = throw new RuntimeException("As array not available for complex updater")
  def updateFromArray(ar: Array[Float]) = throw new RuntimeException("From array not available for complex updater")
  def compress() = this
  def decompress() = this

  def copy() = {
    val sgd = new BasicMMLPSgdUpdater(initialLearningRate)
    sgd.numIterations = this.numIterations
    sgd
  }
  def resetLearningRates(v: Float) = {}
  def compose(u: BasicMMLPSgdUpdater) = this
  def updateWeights(lossGrad: MMLPLossGradient, weights: MMLPWeights): Unit = {
    val eta_t = initialLearningRate / (1.0 + (initialLearningRate * numIterations * lambda))
    weights.wts.addEquals(lossGrad.gr, eta_t.toFloat)
    numIterations += 1
  }
}

class MMLPLossGradient(l: Double, val gr: MMLPLayout) extends LossGradient[MMLPLossGradient](l) {
  def add(other: MMLPLossGradient) = {
    gr.addEquals(other.gr)
    new MMLPLossGradient(l + other.loss, gr)
  }
  def asArray =
    if (gr.length == 1) {
      val pa = gr.get(0)._1.asArray
      val bias = gr.get(0)._2.asArray
      val ar = Array.fill(pa.length + bias.length)(0.0f)
      System.arraycopy(pa, 0, ar, 0, pa.length)
      System.arraycopy(bias, 0, ar, pa.length, bias.length)
      ar
    } else throw new RuntimeException("As Array not feasible with non-linear model")
}

trait Regularizer {
  @inline
  final private def fastSqrt(x: Double) =
    java.lang.Double.longBitsToDouble(((java.lang.Double.doubleToLongBits(x) >> 32) + 1072632448) << 31)


  @inline
  final private def fastSqrt(x: Float) : Float =
    java.lang.Float.intBitsToFloat(532483686 + (java.lang.Float.floatToRawIntBits(x) >> 1))

   /** implementation of scaling - cf. Hinton et al "Improving neural networks by preventing co-adaptation of feature detectors" */
  def rescaleWeightsDense(w_w: DenseMat, w_b: Vec, d1: Int, d2: Int, maxLayerSumSq: Float) = {
    var i = 0; while (i < d1) {
      var ssq = 0.0f
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
  def rescaleWeightsSparse(w_w: SparseMat, w_b: Vec, d1: Int, maxLayerSumSq: Float) = {
    if (maxLayerSumSq > 0.0f) {
    var i = 0; while (i < d1) {
      var ssq = 0.0f
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

  def rescaleWeightsRowSparse(w_w: RowSparseMat, w_b: Vec, d1: Int, maxLayerSumSq: Float) = {
    w_w.forEachRow({(i,row) =>
      var ssq = 0.0f
      var j = 0; while (j < w_w.ncols) {
        ssq += row(j) * row(j)
        j += 1
      }
      val wbi = w_b(i)
      ssq += (wbi * wbi)
      if (ssq > maxLayerSumSq) {
        val sc = fastSqrt(maxLayerSumSq / ssq)
        j = 0; while (j < w_w.ncols) {
          row.update(j, row(j) * sc)
          j += 1
        }
        w_b.update(i, w_b(i) * sc)
      }
      })
  }
}

sealed abstract class UpdaterSpec
case class AdaGradSpec(learnRate: Double = 0.1) extends UpdaterSpec
case class SgdSpec(learnRate: Double = 0.1) extends UpdaterSpec
case class NesterovSgdSpec(learnRate: Double = 0.1) extends UpdaterSpec
case class RMSPropSpec(learnRate: Double = 0.001) extends UpdaterSpec
case object AdaDeltaSpec extends UpdaterSpec

class MMLPAdamUpdater(val alpha: Float, beta1: Float, beta2: Float, val mom1: MMLPLayout, val mom2: MMLPLayout,
                      composeSt: ComposeStrategy = Minimum,
                      maxNormArray: Option[Array[Float]] = None, l1Array: Option[Array[Float]] = None, l2Array: Option[Array[Float]] = None)
  extends Updater[MMLPWeights, MMLPLossGradient, MMLPAdamUpdater] with Regularizer {

  val beta1Inv : Float = (1.0f - beta1)
  val beta2Inv : Float = (1.0f - beta2)
  val epsilon = 1E-8f

  var numIterations = 0
  var beta1T = beta1
  var beta2T = beta2

  val nLayers = mom1.length

  def compress() = this
  def decompress() = this
  def updateFromArray(ar: Array[Float]) = throw new RuntimeException("From array not available for complex updater")
  def asArray : Array[Float] = throw new RuntimeException("As array not available for complex updater")
  def copy() = {
    val sgd = new MMLPAdamUpdater(alpha, beta1, beta2, mom1, mom2, composeSt, maxNormArray, l1Array, l2Array)
    sgd.numIterations = this.numIterations
    sgd.beta1T = this.beta1T
    sgd.beta2T = this.beta2T
    sgd
  }
  def resetLearningRates(v: Float) = {
    mom1.set(v)
    mom2.set(v)
  }

  def compose(u: MMLPAdamUpdater) = {
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

  @inline
  final private def fastSqrt(x: Float) : Float =
    java.lang.Float.intBitsToFloat(532483686 + (java.lang.Float.floatToRawIntBits(x) >> 1))


  def updateWeights(lossGrad: MMLPLossGradient, weights: MMLPWeights): Unit = {
    beta1T *= beta1
    beta2T *= beta2
    for (l <- 0 until nLayers) {
      val l1 = l1Array match {case Some(ar) => ar(l) case None => 0.0f}
      val l2 = l2Array match {case Some(ar) => ar(l) case None => 0.0f}
      val maxNorm = maxNormArray match {case Some(ar) => ar(l) case None => 0.0f}
      val l1Reg = (l1 > 0.0f)
      val l2Reg = (l2 > 0.0f)

      (mom1.get(l), mom2.get(l)) match {
        case ((mom1_w, mom1_b), (mom2_w, mom2_b)) =>
          val (u_w, u_b) = lossGrad.gr.get(l)
          val (w_w, w_b) = weights.wts.get(l)
          val d1 = w_w.getDim1
          val d2 = w_w.getDim2
          u_w match {
            case u_w: DenseMat =>
              var i = 0; while (i < d1) {
                var j = 0; while (j < d2) {
                  val ww = w_w(i,j)
                  val cgr = if (l1Reg) u_w(i, j) - math.signum(ww) * l1 else if (l2Reg) u_w(i,j) - ww * l2 else u_w(i,j)
                  val mt = mom1_w(i, j) * beta1 + (cgr * beta1Inv)
                  val mt_hat = mt / (1.0f - beta1T)
                  mom1_w.update(i, j, mt)
                  val vt = mom2_w(i, j) * beta2 + (cgr * cgr * beta2Inv)
                  val vt_hat = vt / (1.0f - beta2T)
                  mom2_w.update(i, j, vt)
                  w_w.update(i, j, ww + (alpha * mt_hat / (fastSqrt(vt_hat) + alpha)))
                  j += 1
                }
              i += 1
              }
            case u_w: SparseMat =>
              var i = 0; while (i < d1) {
                val row = u_w(i)
                row.forEach { (j, cgr_o) =>
                  val ww = w_w(i,j)
                  val cgr = if (l1Reg) cgr_o - math.signum(ww) * l1 else if (l2Reg) cgr_o - ww * l2 else cgr_o
                  val mt = mom1_w(i, j) * beta1 + (cgr * beta1Inv)
                  val mt_hat = mt / (1.0f - beta1T)
                  mom1_w.update(i, j, mt)
                  val vt = mom2_w(i, j) * beta2 + (cgr * cgr * beta2Inv)
                  val vt_hat = vt / (1.0f - beta2T)
                  mom2_w.update(i, j, vt)
                  w_w.update(i, j, ww + (alpha * mt_hat / (fastSqrt(vt_hat) + alpha)))
                }
                i += 1
              }
            case u_w: RowSparseMat =>
              u_w.forEachRow({(i,row) =>
                val ll = row.getDim
                var j = 0; while (j < ll) {
                  val ww = w_w(i,j)
                  val cgr = if (l1Reg) u_w(i, j) - math.signum(ww) * l1 else if (l2Reg) u_w(i,j) - ww * l2 else u_w(i,j)
                  val mt = mom1_w(i, j) * beta1 + (cgr * beta1Inv)
                  val mt_hat = mt / (1.0f - beta1T)
                  mom1_w.update(i, j, mt)
                  val vt = mom2_w(i, j) * beta2 + (cgr * cgr * beta2Inv)
                  val vt_hat = vt / (1.0f - beta2T)
                  mom2_w.update(i, j, vt)
                  w_w.update(i, j, ww + (alpha * mt_hat / (fastSqrt(vt_hat) + alpha)))
                  j += 1
                }
                })
          }
        u_b match {
          case u_b: SparseVec =>
            u_b.forEach({(i,v) =>
              val cgr = if (l1Reg) u_b(i) - math.signum(w_b(i)) * l1 else if (l2Reg) u_b(i) - w_b(i) * l2 else u_b(i)
              val mt = mom1_b(i) * beta1 + (cgr * beta1Inv)
              val mt_hat = mt / (1.0f - beta1T)
              mom1_b.update(i, mt)
              val vt = mom2_b(i) * beta2 + (cgr * cgr * beta2Inv)
              val vt_hat = vt / (1.0f - beta2T)
              mom2_b.update(i, vt)
              w_b.update(i, w_b(i) + (alpha * mt_hat / (fastSqrt(vt_hat) + alpha)))
              })
          case _ =>
            var i = 0; while (i < d1) {
              val cgr = if (l1Reg) u_b(i) - math.signum(w_b(i)) * l1 else if (l2Reg) u_b(i) - w_b(i) * l2 else u_b(i)
              val mt = mom1_b(i) * beta1 + (cgr * beta1Inv)
              val mt_hat = mt / (1.0f - beta1T)
              mom1_b.update(i, mt)
              val vt = mom2_b(i) * beta2 + (cgr * cgr * beta2Inv)
              val vt_hat = vt / (1.0f - beta2T)
              mom2_b.update(i, vt)
              w_b.update(i, w_b(i) + (alpha * mt_hat / (fastSqrt(vt_hat) + alpha)))
              i += 1
            }
        }
        if (maxNorm > 0.0)
          w_w match {
            case ww: DenseMat => rescaleWeightsDense(ww, w_b, d1, d2, maxNorm)
            case ww: SparseMat => rescaleWeightsSparse(ww, w_b, d1, maxNorm)
            case ww: RowSparseMat => rescaleWeightsRowSparse(ww, w_b, d1, maxNorm)}
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
class MMLPSgdUpdater(val momentum: MMLPLayout, val nesterov: Boolean = true,
                     val initialLearningRate: Float = 0.05f,
                     maxNormArray: Option[Array[Float]] = None,
                     l1Array: Option[Array[Float]] = None,
                     l2Array: Option[Array[Float]] = None, numPoints: Float = 500.0f, compose: ComposeStrategy = Minimum)
  extends Updater[MMLPWeights, MMLPLossGradient, MMLPSgdUpdater] with Regularizer {

  var numIterations = 0
  val nLayers = momentum.length

  def updateFromArray(ar: Array[Float]) = throw new RuntimeException("From array not available for complex updater")
  def asArray : Array[Float] = throw new RuntimeException("As array not available for complex updater")

  def compress() = this
  def decompress() = this

  def copy() = {
    val sgd = new MMLPSgdUpdater(momentum, nesterov, initialLearningRate, maxNormArray, l1Array, l2Array, numPoints, compose)
    sgd.numIterations = this.numIterations
    sgd
  }
  def resetLearningRates(v: Float) = {
    momentum.set(v)
  }

  def compose(u: MMLPSgdUpdater) = {
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

  def updateWeights(lossGrad: MMLPLossGradient, weights: MMLPWeights): Unit = {
    val eta_t = initialLearningRate // / (1.0 + (initialLearningRate * numIterations * lambda))
    numIterations += 1
    val momentumRate = 1.0f - (3.0f / (5.0f + (numIterations.toFloat / numPoints))) // cf. Sutsekver et al. for additional momentum schedules such as:
    // val momentumRate = math.min(0.9, (1.0 - math.pow(2.0,(-1.0 - math.log(math.floor(numIterations * 250 + 1))))))
    for (l <- 0 until nLayers) {
      val l1 = l1Array match {case Some(ar) => ar(l) case None => 0.0f}
      val l2 = l1Array match {case Some(ar) => ar(l) case None => 0.0f}
      val maxNorm = maxNormArray match {case Some(ar) => ar(l) case None => 0.0f}
      val l1Reg = (l1 > 0.0f)
      val l2Reg = (l2 > 0.0f)


      momentum.get(l) match {
        case (mom_w, mom_b) =>
          val (u_w, u_b) = lossGrad.gr.get(l)
          val (w_w, w_b) = weights.wts.get(l)
          val d1 = w_w.getDim1
          val d2 = w_w.getDim2

            u_w match {
              case u_w: DenseMat =>
                var i = 0; while (i < d1) {
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
                i += 1
                }
              case u_w: SparseMat =>
                var i = 0; while (i < d1) {
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
                i += 1
                }
              case u_w: RowSparseMat =>
                u_w.forEachRow({(i,row) =>
                  var j = 0; while (j < d2) {
                  val ww = w_w(i,j)
                  val tv = if (l1Reg) row(j) - math.signum(ww) * l1 else if (l2Reg) row(j) - ww * l2 else row(j)
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
                  })
            }
        u_b match {
          case u_b: DenseVec =>
          var i = 0; while (i < d1) {
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
          case u_b: SparseVec =>
            u_b.forEach({(i,v) =>
              val tv = if (l1Reg) v - math.signum(w_b(i)) * l1 else if (l2Reg) v - w_b(i) * l2 else v
              if (nesterov) {
                val nv = (momentumRate * mom_b(i)) + tv
                mom_b.update(i, nv)
                w_b.update(i, w_b(i) + nv * eta_t)
              } else {
                val tv_a = tv * eta_t
                w_b.update(i, w_b(i) + tv_a + momentumRate * mom_b(i))
                mom_b.update(i, tv_a)
              }
              })
        }
          if (maxNorm > 0.0f)
            w_w match {
              case ww: DenseMat => rescaleWeightsDense(ww, w_b, d1, d2, maxNorm)
              case ww: SparseMat => rescaleWeightsSparse(ww, w_b, d1, maxNorm)
              case ww: RowSparseMat => rescaleWeightsRowSparse(ww, w_b, d1, maxNorm)}
      }
    }
  }
}

class MMLPRMSPropUpdater(val sumSquared: MMLPLayout, val initialLearningRate: Float = 0.9f, lambda: Float = 0.001f,
                         rho: Float = 0.95f, epsilon: Float = 0.003f, maxNormArray: Option[Array[Float]] = None,
                         l1Array: Option[Array[Float]] = None, l2Array: Option[Array[Float]] = None,
                         compose: ComposeStrategy = Minimum)
  extends Updater[MMLPWeights, MMLPLossGradient, MMLPRMSPropUpdater] with Regularizer {

  // set sum squared to initial learning rate
  //sumSquared set initialLearningRate

  val nLayers = sumSquared.length
  var numIterations = 0
  val rhoInv = 1.0f - rho

  def updateFromArray(ar: Array[Float]) = throw new RuntimeException("From array not available for complex updater")
  def asArray : Array[Float] = throw new RuntimeException("As array not available for complex updater")
  def compress() = this
  def decompress() = this


  /*
   * A <i>shallow</i> copy so learning rates are shared across threads/partitions on same machine
   */
  def copy() =
    new MMLPRMSPropUpdater(sumSquared, initialLearningRate, lambda, rho, epsilon, maxNormArray, l1Array, l2Array)

  def resetLearningRates(v: Float) = sumSquared.timesEquals(initialLearningRate * v)

  def compose(u: MMLPRMSPropUpdater) = {
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

  @inline
  final private def fastSqrt(x: Float) : Float =
    java.lang.Float.intBitsToFloat(532483686 + (java.lang.Float.floatToRawIntBits(x) >> 1))


  def updateWeights(lossGrad: MMLPLossGradient, weights: MMLPWeights): Unit = {
    // won't decay the learning rate since this is controled for via RMS update
    val eta_t = initialLearningRate // / (1.0 + (initialLearningRate * numIterations * lambda))
    numIterations += 1
    for (l <- 0 until nLayers) {
      val l1 = l1Array match {case Some(ar) => ar(l) case None => 0.0f}
      val l2 = l2Array match {case Some(ar) => ar(l) case None => 0.0f}
      val maxNorm = maxNormArray match {case Some(ar) => ar(l) case None => 0.0f}
      val l1Reg = (l1 > 0.0f)
      val l2Reg = (l2 > 0.0f)

      sumSquared.get(l) match {
        case (sq_w, sq_b) =>
          val (u_w, u_b) = lossGrad.gr.get(l)
          val (w_w, w_b) = weights.wts.get(l)
          // updater mass gives an additional scalar weight/avg to gradient
          // this is used for mini-batch training
          if (updaterMass != 1.0f) {
            u_w *= updaterMass
            u_b *= updaterMass
          }
          val d1 = w_w.getDim1
          val d2 = w_w.getDim2
            u_w match {
              case u_w: DenseMat =>
                var i = 0; while (i < d1) {
                var j = 0; while (j < d2) {
                  val ww = w_w(i,j)
                  val cgr = if (l1Reg) u_w(i, j) - math.signum(ww).toFloat * l1 else if (l2Reg) u_w(i,j) - ww * l2 else u_w(i,j)
                  val ss = sq_w(i, j) * rho + (cgr * cgr * rhoInv)
                  sq_w.update(i, j, ss)
                  val rms = fastSqrt(ss + epsilon)
                  w_w.update(i, j, ww + (eta_t * cgr / rms))
                  j += 1
                }
                i += 1
                }
              case u_w: SparseMat =>
                var i = 0; while (i < d1) {
                val row = u_w(i)
                row.forEach { (j, cgr_o) =>
                  val ww = w_w(i,j)
                  val cgr = if (l1Reg) cgr_o - math.signum(ww).toFloat * l1 else if (l2Reg) cgr_o - ww * l2 else cgr_o
                  val ss = sq_w(i, j) * rho + (cgr * cgr * rhoInv)
                  sq_w.update(i, j, ss)
                  w_w.update(i, j, ww + (eta_t * cgr / fastSqrt(ss + epsilon)))
                }
                i += 1
                }
              case u_w: RowSparseMat =>
                u_w.forEachRow({(i,row) =>
                  var j = 0; while (j < d2) {
                  val ww = w_w(i,j)
                  val cgr = if (l1Reg) u_w(i, j) - math.signum(ww) * l1 else if (l2Reg) u_w(i,j) - ww * l2 else u_w(i,j)
                  val ss = sq_w(i, j) * rho + (cgr * cgr * rhoInv)
                  sq_w.update(i, j, ss)
                  val rms = fastSqrt(ss + epsilon)
                  w_w.update(i, j, ww + (eta_t * cgr / rms))
                  j += 1
                  }
               })
            }
        u_b match {
          case u_b: SparseVec =>
            u_b.forEach({(i,v) =>
              val cgr = if (l1Reg) v - math.signum(w_b(i)) * l1 else if (l2Reg) v - w_b(i) * l2 else v
              val ss = sq_b(i) * rho + (cgr * cgr * rhoInv)
              sq_b.update(i, ss)
              val rms = fastSqrt(ss + epsilon)
              w_b.update(i, w_b(i) + (eta_t * cgr / rms))
              })
          case u_b: DenseVec =>
            var i = 0; while (i < d1) {
            val cgr = if (l1Reg) u_b(i) - math.signum(w_b(i)) * l1 else if (l2Reg) u_b(i) - w_b(i) * l2 else u_b(i)
            val ss = sq_b(i) * rho + (cgr * cgr * rhoInv)
            sq_b.update(i, ss)
            val rms = fastSqrt(ss + epsilon)
            w_b.update(i, w_b(i) + (eta_t * cgr / rms))
            i += 1
          }
        }
        if (maxNorm > 0.0f)
            w_w match {
              case ww: DenseMat => rescaleWeightsDense(ww, w_b, d1, d2, maxNorm)
              case ww: SparseMat => rescaleWeightsSparse(ww, w_b, d1, maxNorm)
              case ww: RowSparseMat => rescaleWeightsRowSparse(ww, w_b, d1, maxNorm) }
      }
    }
  }
}

class MMLPAdaGradUpdater(val sumSquared: MMLPLayout, val initialLearningRate: Float = 0.9f, maxNormArray: Option[Array[Float]] = None,
                         l1Array: Option[Array[Float]] = None, l2Array: Option[Array[Float]] = None,
                         compose: ComposeStrategy = Minimum)
  extends Updater[MMLPWeights, MMLPLossGradient, MMLPAdaGradUpdater] with Regularizer {

  sumSquared set initialLearningRate // set sum squared to initial learning rate

  val nLayers = sumSquared.length

  def asArray : Array[Float] = throw new RuntimeException("As array not available for complex updater")
  def updateFromArray(ar: Array[Float]) = throw new RuntimeException("From array not available for complex updater")
  def compress() = this
  def decompress() = this

  /*
   * A <i>shallow</i> copy so learning rates are shared across threads/partitions
   */
  def copy() = new MMLPAdaGradUpdater(sumSquared.copy(), initialLearningRate, maxNormArray, l1Array, l2Array, compose)

  def resetLearningRates(v: Float) = sumSquared.timesEquals(initialLearningRate * v)

  def compose(u: MMLPAdaGradUpdater) = {
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


  @inline
  final private def fastSqrt(x: Float) : Float =
    java.lang.Float.intBitsToFloat(532483686 + (java.lang.Float.floatToRawIntBits(x) >> 1))

  def updateWeights(lossGrad: MMLPLossGradient, weights: MMLPWeights): Unit = {
    for (l <- 0 until nLayers) {
      val l1 = l1Array match {case Some(ar) => ar(l) case None => 0.0f}
      val l2 = l2Array match {case Some(ar) => ar(l) case None => 0.0f}
      val maxNorm = maxNormArray match {case Some(ar) => ar(l) case None => 0.0f}
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
            u_w match {
              case u_w: DenseMat =>
                var i = 0; while (i < d1) {
                var j = 0; while (j < d2) {
                  val ww = w_w(i, j)
                  val cgr = if (l1Reg) u_w(i, j) - math.signum(ww) * l1 else if (l2Reg) u_w(i,j) - ww * l2 else u_w(i,j)
                  val csq = sq_w(i, j)
                  val nsq = csq + (cgr * cgr)
                  sq_w.update(i, j, nsq)
                  w_w.update(i, j, ww + (initialLearningRate * cgr / (initialLearningRate + fastSqrt(nsq))))
                  j += 1
                }
                i += 1
                }
              case u_w: SparseMat =>
                var i = 0; while (i < d1) {
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
                i += 1
                }
              case u_w: RowSparseMat =>
                u_w.forEachRow({(i,row) =>
                  var j = 0; while (j < d2) {
                  val ww = w_w(i,j)
                  val cgr = if (l1Reg) row(j) - math.signum(ww) * l1 else if (l2Reg) row(j) - ww * l2 else row(j)
                  val csq = sq_w(i, j)
                  val nsq = csq + (cgr * cgr)
                  sq_w.update(i, j, nsq)
                  w_w.update(i, j, ww + (initialLearningRate * cgr / (initialLearningRate + fastSqrt(nsq))))
                  j += 1
                }
              })
            }
          u_b match {
            case u_b: DenseVec =>
              var i = 0; while (i < d1) {
            val csq = sq_b(i)
            val cgr = if (l1Reg) u_b(i) - math.signum(w_b(i)) * l1 else if (l2Reg) u_b(i) - w_b(i) * l2 else u_b(i)
            val nsq = csq + (cgr * cgr)
            sq_b.update(i, nsq)
            val cw = w_b(i)
            w_b.update(i, cw + (initialLearningRate * cgr / (initialLearningRate + fastSqrt(nsq))))
            i += 1
          }
            case u_b: SparseVec =>
              u_b.forEach({(i,v) =>
                val csq = sq_b(i)
                val cgr = if (l1Reg) v - math.signum(w_b(i)) * l1 else if (l2Reg) v - w_b(i) * l2 else v
                val nsq = csq + (cgr * cgr)
                sq_b.update(i, nsq)
                val cw = w_b(i)
                w_b.update(i, cw + (initialLearningRate * cgr / (initialLearningRate + fastSqrt(nsq))))
              })
          }
          if (maxNorm > 0.0) w_w match {
            case ww: DenseMat => rescaleWeightsDense(ww, w_b, d1, d2, maxNorm)
            case ww: SparseMat => rescaleWeightsSparse(ww, w_b, d1, maxNorm)
            case ww: RowSparseMat => rescaleWeightsRowSparse(ww, w_b, d1, maxNorm)}
      }
    }
  }
}

class MMLPAdaDeltaUpdater(val sumSquared: MMLPLayout, val prevUpdates: MMLPLayout, val epsilon: Float = 0.001f, val rho: Float = 0.95f,
                          maxLayerSumSq: Float = 10.0f, compose: ComposeStrategy = Minimum, maxNorm: Boolean = true)
  extends Updater[MMLPWeights, MMLPLossGradient, MMLPAdaDeltaUpdater] with Regularizer {

  sumSquared set epsilon // set sum squared to epsilon

  val nLayers = sumSquared.length
  val rhoInv = 1.0f - rho

  @inline
  final private def fastSqrt(x: Double) =
    java.lang.Double.longBitsToDouble(((java.lang.Double.doubleToLongBits(x) >> 32) + 1072632448) << 31)

  @inline
  final private def fastSqrt(x: Float) : Float =
    java.lang.Float.intBitsToFloat(532483686 + (java.lang.Float.floatToRawIntBits(x) >> 1))

  def compress() = this
  def decompress() = this

  def updateFromArray(ar: Array[Float]) = throw new RuntimeException("From array not available for complex updater")
  def asArray : Array[Float] = throw new RuntimeException("As array not available for complex updater")
  /*
   * A <i>shallow</i> copy so learning rates are shared across threads/partitions
   */
  def copy() = new MMLPAdaDeltaUpdater(sumSquared, prevUpdates, epsilon, rho)

  def resetLearningRates(v: Float) = {
    sumSquared.set(v)
    prevUpdates.set(v)
  }

  def compose(u: MMLPAdaDeltaUpdater) = {
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

  def updateWeights(lossGrad: MMLPLossGradient, weights: MMLPWeights): Unit = {
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

abstract class MMLPPredictor[R,C <: Confusion[C]](val network: ANNetwork, val getScores: Boolean = false)
  extends EvalPredictor[MMLPFactor, MMLPWeights, R, C] with Serializable {

  def getLoss(u: MMLPFactor, w: MMLPWeights): Double = {
    network.forwardPass(u.getInput, u.getOutput, w, false)
    network.getCost
  }

}

class CategoricalMMLPPredictor(n: ANNetwork, gs: Boolean = false)
  extends MMLPPredictor[Int, DiscreteConfusion](n,gs) with Serializable {

  def getPrediction(u: MMLPFactor, w: MMLPWeights): Int = {
    network.forwardPass(u.getInput, u.getOutput, w, false)
    val uscores = network.outLayer.getOutput(false)
    uscores.argmax
  }

  def getScoredPredictions(u: MMLPFactor, w: MMLPWeights): Seq[(Float, Int)] = {
    network.forwardPass(u.getInput, u.getOutput, w, false)
    val uscores = network.outLayer.getOutput(false) // always softmax these scores
    uscores.asArray.toSeq.zipWithIndex
  }

  def getConfusion(u: MMLPFactor, w: MMLPWeights): DiscreteConfusion = {
    val p = getPrediction(u, w)
    val ncats = network.outLayer.getOutput(false).getDim
    val targetVal = u.getOutput.argmax // output target is output with highest score (usually one-hot)
    if (getScores) {
      val scores = network.outLayer.getOutput(false).copy.asArray
      DiscreteConfusion(scores.length, p, targetVal, scores)
    } else DiscreteConfusion(ncats, p, targetVal)
  }
}

class RegressionMMLPPredictor(n: ANNetwork, gs: Boolean = false)
  extends MMLPPredictor[Vec, RegressionConfusion](n, gs) {

  def getPrediction(u: MMLPFactor, w: MMLPWeights): Vec = {
    network.forwardPass(u.getInput, u.getOutput, w, false)
    network.outLayer.getOutput(false)
  }

  def getScoredPredictions(u: MMLPFactor, w: MMLPWeights) = throw new RuntimeException("Not implemented")
  def getConfusion(u: MMLPFactor, w: MMLPWeights) : RegressionConfusion = throw new RuntimeException("Not IMplemented")
}


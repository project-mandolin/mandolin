package org.mitre.mandolin.predict

/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.optimize.Weights
import collection.mutable.ArrayBuffer

/**
  * Functionality for predictors; they need to provide
  * a method that assigns a prediction to a (test/training) unit given the model parameters.
  */
abstract class Predictor[U, W <: Weights[W], +R] {
  def getPrediction(unit: U, weights: W): R

  def getScoredPredictions(unit: U, weights: W): Seq[(Float, R)]
}

/**
  * Predictor that provides the loss for an item as well as a confusion used for scoring.
  */
abstract class EvalPredictor[U, W <: Weights[W], +R, C <: Confusion[C]] extends Predictor[U, W, R] {
  def getLoss(unit: U, weights: W): Double

  def getConfusion(unit: U, weights: W): C
}

/**
  * Generalized notion of 'confusion' that captures error information obtained from applying
  * the classifier to a given test/training unit.  In simplest, case this can just be an accuracy value.
  * More generally, it may be a full confusion matrix.
  */
abstract class Confusion[C <: Confusion[C]] {
  def compose(c: C): C

  def getMatrix: ConfusionMatrix

  def getAreaUnderROC(i: Int): Double

  def getTotalAreaUnderROC(): Double

  def getAccuracyAtThroughPut(th: Double): Double

  def getROCofIndex(i: Int): (Array[(Double, Double, Double)], Int)
}

/**
  * Just captures SSE
  */
class RegressionConfusion(var error: Double) extends Confusion[RegressionConfusion] {
  def this(v1: Double, v2: Double) = this((v1 - v2) * (v1 - v2))

  def compose(other: RegressionConfusion) = {
    this.error += other.error
    this
  }

  def getAreaUnderROC(i: Int) = 0.0

  def getTotalAreaUnderROC() = 0.0

  def getAccuracyAtThroughPut(th: Double) = 0.0

  def getMatrix = new ConfusionMatrix(error.toFloat)

  def getROCofIndex(i: Int) = throw new RuntimeException("Unsupported method")
}

class DiscreteConfusion(val dim: Int, val matrix: Array[Array[Float]], val scoreVals: ArrayBuffer[(Array[Float], Int)], var weight: Int = -1)
  extends Confusion[DiscreteConfusion] with Serializable {

  def this(dim: Int) = this(dim, Array.tabulate(dim) { _ => Array.fill(dim)(0.0f) }, new ArrayBuffer[(Array[Float], Int)]())

  def this(dim: Int, m: Array[Array[Float]]) = this(dim, m, new ArrayBuffer[(Array[Float], Int)]())

  override def toString() = {
    val sbuf = new StringBuilder
    scoreVals foreach { case (s, g) =>
      s foreach { v => sbuf append (" " + v) }
      sbuf append ("   with g = " + g + "\n")
    }
    sbuf.toString
  }


  def compose(other: DiscreteConfusion) = {
    if (weight < 0) {
      var i = 0;
      while (i < dim) {
        var j = 0;
        while (j < dim) {
          matrix(i)(j) += other.matrix(i)(j)
          j += 1
        }
        i += 1
      }
      scoreVals ++= other.scoreVals
      this
    } else {
      val nweight = weight + other.weight
      val curSc = matrix(0)(0)
      val sc = ((weight * curSc) + (other.matrix(0)(0) * other.weight)) / nweight.toFloat
      matrix(0)(0) = sc
      weight = nweight
      this
    }
  }

  def getAccuracyAtThroughPut(th: Double): Double = {
    val itemsWithTopScore = (scoreVals map { case (sc, g) => (sc.max, sc, g) }).toVector.sortWith { case (t1, t2) => t1._1 > t2._1 }
    val splitInt = (itemsWithTopScore.length * th).toInt
    val sortedTestItems = itemsWithTopScore.slice(0, splitInt)
    var cor = 0
    sortedTestItems foreach { case (mx, scs, g) => if (scs(g) >= mx) cor += 1 }
    cor.toDouble / splitInt
  }

  def getROCofIndex(i: Int): (Array[(Double, Double, Double)], Int) = {
    val sBuf = new ArrayBuffer[(Double, Int)]
    scoreVals foreach { case (sc, gold) => sBuf append ((sc(i), if (gold == i) 1 else 0)) }
    val sorted = sBuf.sortWith((a, b) => a._1 > b._1) // descending order
    val total = sorted.length
    val pos = sorted.filter(_._2 == 1).length
    val neg = total - pos
    var tp = 0
    var fp = 0
    val roc = sorted map {
      case (sc, g) =>
        if (g == 1) tp += 1 else fp += 1
        val tpr = if (pos > 0.0) tp.toDouble / pos else 0.0
        val fpr = if (neg > 0.0) fp.toDouble / neg else 0.0
        (tpr, fpr, sc)
    }
    (roc.toArray, pos)
  }

  def getAreaUnderROC(i: Int): Double = {
    if (weight < 0) {
      val (roc, pos) = getROCofIndex(i)
      var area = 0.0
      roc.sliding(2).foreach { pair =>
        area += ((pair(1)._2 - pair(0)._2) * (pair(1)._1 + pair(0)._1) / 2.0) // trapezoid area
      }
      area
    } else 0.0
  }

  def getAreaUnderROCWithPositiveWeight(i: Int): (Double, Int) = {
    val (roc, pos) = getROCofIndex(i)
    var area = 0.0
    roc.sliding(2).foreach { pair =>
      area += ((pair(1)._2 - pair(0)._2) * (pair(1)._1 + pair(0)._1) / 2.0) // trapezoid area
    }
    (area, pos)
  }

  /*
   * This computes an Area under the ROC for multi-class problems by simply taking
   * a weighted average of the au-ROC for each class, following Provost and Domingos 2001.
   * An alternative is the method of Hand and Till 2001, which computes curves over all
   * pairs of classes.
   */
  def getTotalAreaUnderROC(): Double = {
    var sum = 0.0
    var tot = 0
    for (i <- 0 until dim) {
      val (area, pw) = getAreaUnderROCWithPositiveWeight(i)
      tot += pw
      sum += area * pw
    }
    if (tot > 0.0f) sum / tot else 0.0
  }

  def getMatrix = new ConfusionMatrix(matrix)

}

object DiscreteConfusion {
  def apply(dim: Int, pairs: Vector[(Int, Int)]) = {
    if (dim < 100) {
      val m = Array.tabulate(dim)(_ => Array.fill(dim)(0.0f))
      pairs foreach { case (p, g) => m(p)(g) += 1 }
      new DiscreteConfusion(dim, m)
    } else {
      var sc = 0.0f
      pairs foreach { case (p, g) => if (p == g) sc += 1.0f }
      val n = pairs.length
      new DiscreteConfusion(dim, Array(Array(sc / n)), ArrayBuffer(), n)
    }
  }

  def apply(dim: Int, pred: Int, gold: Int) = {
    if (dim < 100) {
      val m = Array.tabulate(dim)(_ => Array.fill(dim)(0.0f))
      m(pred)(gold) = 1.0f
      new DiscreteConfusion(dim, m)
    } else {
      val sc = if (pred == gold) 1.0f else 0.0f
      new DiscreteConfusion(dim, Array(Array(sc)), ArrayBuffer(), 1)
    }
  }

  def apply(dim: Int, pred: Int, gold: Int, scores: Array[Float]) = {
    if (dim < 100) {
      val m = Array.tabulate(dim)(_ => Array.fill(dim)(0.0f))
      m(pred)(gold) = 1.0f
      new DiscreteConfusion(dim, m, ArrayBuffer((scores, gold)))
    } else {
      val sc = if (pred == gold) 1.0f else 0.0f
      new DiscreteConfusion(dim, Array(Array(sc)), ArrayBuffer(), 1)
    }
  }
}

class ConfusionMatrix(val m: Array[Array[Float]]) {

  def this(v: Float) = this(Array(Array(v)))

  val dim = m.length

  lazy val total: Double = {
    var t = 0.0
    for (i <- 0 until dim; j <- 0 until dim) t += m(i)(j)
    t.toDouble
  }

  def getAccuracy: Double = {
    if (m.length < 2) m(0)(0) else {
      var correct = 0.0
      for (i <- 0 until dim) correct += m(i)(i)
      correct / total
    }
  }

  def ++(cm: ConfusionMatrix): ConfusionMatrix = {
    assert(dim == cm.dim)
    for (i <- 0 until dim; j <- 0 until dim) m(i)(j) += cm.m(i)(j)
    this
  }

  def prettyPrint(categories: Array[String], os: java.io.PrintStream): Unit = {
    assert(categories.length == dim)
    os.print("\t")
    for (i <- 0 until dim) os.print("\t%s".format(categories(i)))

    for (i <- 0 until dim) {
      os.print("\n\t%s".format(categories(i)))
      for (j <- 0 until dim) {
        os.print("\t%.1f".format(m(i)(j)))
      }
    }
  }

  def prettyPrint(os: java.io.PrintStream): Unit = prettyPrint(Array.tabulate(dim)(_.toString), os)

  def prettyPrint(): Unit = {
    prettyPrint(System.out)
    System.out.flush()
  }

  def prettyPrint(f: java.io.File): Unit = {
    val os = new java.io.PrintStream(f)
    prettyPrint(os)
    os.close()
  }

  def prettyPrint(categories: Array[String], f: java.io.File): Unit = {
    val os = new java.io.PrintStream(f)
    prettyPrint(categories, os)
    os.close()
  }

}


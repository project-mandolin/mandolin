package org.mitre.mandolin.transform
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.gm.{ Feature, NonUnitFeature }

class FeatureImportance {

  private def log2(v: Double) = math.log(v) / math.log(2.0)

  private def countValue(ar: Array[Int], dim: Int, vl: Int) = {
    var i = 0
    var sum = 0
    while (i < dim) {
      if (ar(i) == vl) sum += 1
      i += 1
    }
    sum
  }

  private def jointCount(ar1: Array[Int], ar2: Array[Int], ar1Val: Int, dim: Int) = {
    var i = 0
    var cnt = 0
    while (i < dim) {
      if ((ar1(i) == ar1Val) && ar2(i) > 0) cnt += 1
      i += 1
    }
    cnt
  }

  def empiricalMutualInfo(ys: Array[Int], yValCounts: Array[Int], xs: Array[Int]) = {
    var mi = 0.0
    val dim = xs.length
    val xsum = countValue(xs, dim, 1)
    var jcnts = List[(Int, Int)]()
    val perClassEntropies =
      for (i <- 0 until yValCounts.length) yield {
        val jcnt = jointCount(ys, xs, i, dim)
        val jointProb = (jcnt.toDouble / dim)
        val xProb = xsum.toDouble / dim
        val yProb = yValCounts(i).toDouble / dim
        jcnts = (i, jcnt) :: jcnts
        val pmi = if ((jointProb > 0.0) && (xProb > 0.0) && (yProb > 0.0)) log2((jointProb) / (xProb * yProb)) else 0.0
        mi += (jointProb * pmi)
        (pmi / -(log2(math.max(xProb, yProb))))
      }
    (mi, xsum, jcnts, perClassEntropies)
  }

  def computeMutualInfo(dataArr: Array[Array[Feature]], yvs: Array[Int], nfs: Int, nls: Int, topK: Int) = {
    val dl = dataArr.length
    val yValCnts: Array[Int] = (for (i <- 0 until nls) yield countValue(yvs, dl, i)).toArray
    val mutualInfoRecords = new collection.mutable.ListBuffer[(Double, Int, Int, List[(Int, Int)], IndexedSeq[Double])]()
    var k = 0; while (k < nfs) {
      val ar = Array.fill(dl)(0)
      var i = 0; while ((i < dl)) { // && !nonBinary) {
        dataArr(i) foreach { f => if (f.fid == k) ar(i) = 1 } // if ((f.value < 1.00001) && (f.value > 0.999999)) ar(i) = 1 else nonBinary = true}
        i += 1
      }
      val (mi, xsum, jcnts, perLabelEntropies) = empiricalMutualInfo(yvs, yValCnts, ar)
      mutualInfoRecords append ((mi, k, xsum, jcnts, perLabelEntropies))
      k += 1
    }
    val sorted = mutualInfoRecords.toIndexedSeq.sortWith((a,b) => a._1 > b._1) // sort by overall MI score
    sorted.slice(0,topK) map {_._2}    
  }

}

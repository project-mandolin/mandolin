package org.mitre.mandolin.mlp

/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.config._
import org.mitre.mandolin.util.{Alphabet, IOAssistant}
import org.mitre.mandolin.predict.OutputConstructor
import com.typesafe.config.Config
import net.ceedubs.ficus.Ficus._


/**
  * Output constructor for MMLP models. Writes a vector of outputs
  * which may form a distribution, e.g. with a SoftMax output layer.
  * Also includes comments/metadata from corresponding inputs so outputs
  * can be linked back to the data from which they came.
  *
  * @author wellner
  */
class MMLPPosteriorOutputConstructor extends OutputConstructor[String, Seq[(Float, Int)], String] {
  def constructOutput(i: String, r: Seq[(Float, Int)], s: String): String = {
    val (vec, id) = i.split('#').toList match {
      case a :: b :: Nil => (a, Some(b))
      case a :: _ => (a, None)
      case Nil => throw new RuntimeException("Invalid line: " + i)
    }
    val idStr = id.getOrElse("-1")
    val resStr = responseString(r)
    val sbuf = new StringBuilder
    sbuf append idStr
    sbuf append resStr
    sbuf.toString
  }

  def responseString(r: Seq[(Float, Int)]): String = {
    val sortedByY = r.sortWith((a, b) => a._2 < b._2)
    val sbuf = new StringBuilder
    sortedByY foreach {
      case (d, i) =>
        sbuf append ','
        sbuf append d.toString
    }
    sbuf.toString
  }

  def intToResponseString(r: Int): String = "UNK"
}

class MMLPMostProbableOutputConstructor extends OutputConstructor[String, Int, String] {
  def constructOutput(i: String, r: Int, s: String): String = {
    val (_, id) = i.split('#').toList match {
      case a :: b :: Nil => (a, Some(b))
      case a :: _ => (a, None)
      case Nil => throw new RuntimeException("Invalid line: " + i)
    }
    val idStr = id.getOrElse("-1")
    val resStr = responseString(r)
    val sbuf = new StringBuilder
    sbuf append idStr
    sbuf append resStr
    sbuf.toString
  }

  def responseString(r: Int): String = {
    r.toString
  }

  def intToResponseString(r: Int): String = "UNK"
}

class MMLPRegressionOutputConstructor extends OutputConstructor[String, (Double, Double), String] {
  def constructOutput(i: String, r: (Double, Double), s: String): String = {
    val (vec, id) = i.split('#').toList match {
      case a :: b :: Nil => (a, Some(b))
      case a :: _ => (a, None)
      case Nil => throw new RuntimeException("Invalid line: " + i)
    }
    val idStr = id.getOrElse("-1")
    val resStr = responseString(r)
    val sbuf = new StringBuilder
    sbuf append idStr
    sbuf append resStr
    sbuf.toString
  }

  def responseString(r: (Double, Double)): String = {
    r._1.toString + ", " + r._2.toString
  }

  def intToResponseString(r: Int): String = "UNK"
}


/**
  * A separate utility program that takes inputs in a Sparse vector representation
  * and maps them to a tab-separate file dense representation with the feature
  * names (in the sparse vector input file) as column headers. The first column is
  * the label, which is the original string representation of the label in the
  * sparse input file.
  *
  * @author wellner
  */
object DenseVectorWriter {

  def exportVectorsDense(f: String, io: IOAssistant, fvs: Vector[MMLPFactor], alphabet: Alphabet, la: Alphabet) = {
    val os = io.getPrintWriterFor(f, false)
    val invMap = la.getInverseMapping
    val im = alphabet.getInverseMapping.toList.sortWith { case ((k1, v1), (k2, v2)) => k1 < k2 }
    val ar = im.toArray
    os.write("key")
    os.write('\t')
    os.write("label")
    for (i <- 0 until ar.length) {
      os.write('\t')
      os.write(ar(i)._2.toString)
    }
    os.write('\n')
    fvs foreach { l =>
      val dv = l.getInput
      val outInd = l.getOneHot
      val outLab = invMap(outInd)
      l.getUniqueKey foreach { k =>
        os.write(k)
        os.write('\t')
      }
      os.write(outLab)
      for (i <- 0 until dv.getSize) {
        os.write('\t')
        os.write(dv(i).toString)
      }
      os.write('\n')
    }
    os.close()
  }

  def exportVectorsDense(f: java.io.File, fvs: Vector[MMLPFactor], alphabet: Alphabet, la: Alphabet) = {
    val os = new java.io.PrintWriter(f)
    val invMap = la.getInverseMapping
    val im = alphabet.getInverseMapping.toList.sortWith { case ((k1, v1), (k2, v2)) => k1 < k2 }
    val ar = im.toArray
    os.write("key")
    os.write('\t')
    os.write("label")
    for (i <- 0 until ar.length) {
      os.write('\t')
      os.write(ar(i)._2.toString)
    }
    os.write('\n')
    fvs foreach { l =>
      val dv = l.getInput
      val outInd = l.getOneHot
      val outLab = invMap(outInd)
      l.getUniqueKey foreach { k =>
        os.write(k)
        os.write('\t')
      }
      os.write(outLab)
      for (i <- 0 until dv.getSize) {
        os.write('\t')
        os.write(dv(i).toString)
      }
      os.write('\n')
    }
    os.close()
  }
}

package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.util.DenseTensor1
import org.mitre.mandolin.config.{ LearnerSettings, DeepNetSettings, OnlineLearnerSettings, BatchLearnerSettings, DecoderSettings, AppConfig }
import org.mitre.mandolin.util.{ RandomAlphabet, StdAlphabet, IdentityAlphabet, Alphabet, AlphabetWithUnitScaling, IOAssistant }
import org.mitre.mandolin.predict.OutputConstructor
import org.mitre.mandolin.glp.local.LocalProcessor
import org.mitre.mandolin.glp.spark.DistributedProcessor
import scala.reflect.ClassTag

class GLPModelSettings(args: Array[String]) extends LearnerSettings(args) 
  with OnlineLearnerSettings with BatchLearnerSettings with DecoderSettings with DeepNetSettings

/**
 * Output constructor for GLP models. Writes a vector of outputs
 * which may form a distribution, e.g. with a SoftMax output layer.
 * Also includes comments/metadata from corresponding inputs so outputs
 * can be linked back to the data from which they came.
 * @author wellner
 */
class GLPPosteriorOutputConstructor extends OutputConstructor[String, Seq[(Double, Int)], String] {
  def constructOutput(i: String, r: Seq[(Double, Int)], s: String): String = {
    val (vec, id) = i.split('#').toList match {
      case a :: b :: Nil => (a, Some(b))
      case a :: _        => (a, None)
      case Nil           => throw new RuntimeException("Invalid line: " + i)
    }
    val idStr = id.getOrElse("-1")
    val resStr = responseString(r)
    val sbuf = new StringBuilder
    sbuf append idStr
    sbuf append resStr
    sbuf.toString
  }

  def responseString(r: Seq[(Double, Int)]): String = {
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


/**
 * A separate utility program that takes inputs in a Sparse vector representation 
 * and maps them to a tab-separate file dense representation with the feature
 * names (in the sparse vector input file) as column headers. The first column is 
 * the label, which is the original string representation of the label in the 
 * sparse input file.
 * @author wellner
 */
object DenseVectorWriter extends org.mitre.mandolin.app.AppMain {
  
  def exportVectorsDense(f: String, io: IOAssistant, fvs: Vector[GLPFactor], alphabet: Alphabet, la: Alphabet) = {
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
      l.getUniqueKey foreach {k =>
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
  
  def exportVectorsDense(f: java.io.File, fvs: Vector[GLPFactor], alphabet: Alphabet, la: Alphabet) = {
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
      l.getUniqueKey foreach {k =>
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
  
  def main(args: Array[String]) : Unit = {
    /*
    val appSettings = new GLPModelSettings(args)
    val lines = scala.io.Source.fromFile(appSettings.trainFile).getLines().toVector
    val localProcessor = new LocalProcessor(appSettings)
    val (_,_,_,fe,fa,la) = localProcessor.getFullComponents()
    appSettings.denseOutputFile foreach { nf =>
        val f = new java.io.File(nf)        
        val fvecs = lines map { l => fe.extractFeatures(l) }
        exportVectorsDense(f, fvecs, fa, la)
      }
      *
      */
  }
} 



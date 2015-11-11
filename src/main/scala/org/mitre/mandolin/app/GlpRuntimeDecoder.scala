package org.mitre.mandolin.app
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.glp.{StdGLPFactor, GLPPredictor, GLPPosteriorOutputConstructor}
import org.mitre.mandolin.glp.local.LocalGLPModelReader
import org.mitre.mandolin.gm.NonUnitFeature
import org.mitre.mandolin.util.{IOAssistant, DenseTensor1 => DenseVec, LocalIOAssistant}
import scala.collection.JavaConversions._

case class StringDoublePair(val str: String, val value: Double)

/**
 * A simple runtime decoder for trained GLPs. Provides a lightweight way to use
 * a trained classifier from Scala or Java by providing sparse feature vectors as a simple
 * string or list of strings.  Short of a full Java API, this provides a simple method
 * that takes in a Java or Scala list of strings as a sparse feature representation and
 * returns a single floating point corresponding to the posterior probability associated
 * with the positive class.
 * @param filePath path to input file containing serialized model
 * @author wellner
 */
class GlpRuntimeDecoder(filePath: String, io: IOAssistant, posCase: String) {
  def this(filePath: String, io: IOAssistant) = this(filePath, io, "")
  def this(filePath: String) = this(filePath, new LocalIOAssistant, "")
  
  val reader = new LocalGLPModelReader
  val model = reader.readModel(filePath, io)
  
  val posIndex = model.la.ofString(posCase)
  
  val fe = model.fe
  val fa = fe.getAlphabet
  val laSize = model.la.getSize
  val invLa  = model.la.getInverseMapping
  
  val predictor = new GLPPredictor(model.evaluator.glp, true)
  val evalDecoder = new org.mitre.mandolin.predict.local.LocalPosteriorDecoder(fe, predictor, new GLPPosteriorOutputConstructor)

  fa.ensureFixed // make sure the feature alphabet is fixed
  
  var unknownFeatures = Set[String]()
  
  /** @return A Java Set of Strings containing all features provided that weren't recognized by the model alphabet */
  def getUnknownFeatures() : java.util.Set[String] = {
    unknownFeatures
  }
  
  /** Resets the buffer holding the unknown features */
  def resetUnknownFeatures() : Unit = {
    unknownFeatures = Set[String]()
  }
  
  def positivePosterior(s: java.util.List[String]) : java.lang.Double = {
    val scalaList : scala.collection.mutable.Buffer[String] = s
    positivePosterior(scalaList.toList)  
  }
  
  def positivePosteriorFromStringDoublePairs(s: java.util.List[StringDoublePair]) : java.lang.Double = {
    val scalaList : scala.collection.mutable.Buffer[StringDoublePair] = s
    positivePosterior(scalaList.toList)
  }
  
  def positivePosterior(s: List[StringDoublePair]) : java.lang.Double = {
    val scores = decodePairs(s)
    (scores.find{case (s,i) => i == posIndex}).get._2
  }
  
  def positivePosterior(s: List[String]) = {
    val scores = decode(s)
    (scores.find{case (s,i) => i == posIndex}).get._2
  }
  
  def decode(s: String) = {
    val factor = fe.extractFeatures(s)
    predictor.getScoredPredictions(factor, model.wts)
  }
  
  def decode(li: List[String]) = {
    val dVec : DenseVec = DenseVec.zeros(fa.getSize)
    li foreach {s =>
      val fid = fa.ofString(s)
      if (fid >= 0) dVec(fa.ofString(s)) = 1.0 else unknownFeatures += s}
    val lv = DenseVec.zeros(laSize)
    val factor = new StdGLPFactor(-1, dVec, lv, None)
    predictor.getScoredPredictions(factor, model.wts)    
  }
  
  def decodePairs(li: List[StringDoublePair]) = {
    val dVec : DenseVec = DenseVec.zeros(fa.getSize)
    li foreach {s =>
      val fid = fa.ofString(s.str)
      if ((fid >= 0) && (fid < fa.getSize)) dVec(fa.ofString(s.str)) = s.value else unknownFeatures += s.str}
    val lv = DenseVec.zeros(laSize)
    val factor = new StdGLPFactor(-1, dVec, lv, None)
    predictor.getScoredPredictions(factor, model.wts)
  }
  
  private def getBest(r: Seq[(Double,Int)]) : (Int,Double) = {
    var bi = 0
    var bv = 0.0
    r foreach {case (v,i) => if (v > bv) {bv = v; bi = i}}
    (bi,bv)
  }
  
  def decodeMPE(s: String) = {
    val res = decode(s)
    getBest(res)._1
  }
  
  def decodePairsMPE(li: java.util.List[StringDoublePair]) : Int = {
    val scalaList : scala.collection.mutable.Buffer[StringDoublePair] = li
    decodePairsMPE(scalaList.toList)
  }
  
  def decodePairsMPE(li: List[StringDoublePair]) : Int = {
    val res = decodePairs(li)
    getBest(res)._1
  }
  
  def decodePairsMPEWithScore(li: List[StringDoublePair]) : StringDoublePair = {
    val res = decodePairs(li)
    val best = getBest(res)
    val bestLabel = invLa(best._1)
    new StringDoublePair(bestLabel, best._2)
  }
  
  def decodePairsMPEWithScore(li: java.util.List[StringDoublePair]) : StringDoublePair = {
    val scalaList : scala.collection.mutable.Buffer[StringDoublePair] = li
    decodePairsMPEWithScore(scalaList.toList)
  }
}
package org.mitre.mandolin.xg

import org.mitre.mandolin.util.{DenseTensor1 => DenseVec, LocalIOAssistant, Tensor2 => Mat,IOAssistant}
import org.mitre.mandolin.app.{StringDoublePair, RuntimeDecoder}
import org.mitre.mandolin.mlp.StdMMLPFactor
import scala.collection.JavaConversions._
import ml.dmlc.xgboost4j.scala.{XGBoost, DMatrix}

class XGBinaryRuntimeDecoder(filePath: String, io: IOAssistant, posCase: String) extends RuntimeDecoder with DataConversions {
  def this(filePath: String, io: IOAssistant) = this(filePath, io, "")
  def this(filePath: String) = this(filePath, new LocalIOAssistant, "")
  
  val reader = new StandaloneXGBoostModelReader
  
  val spec = reader.readModel(filePath, io)  
  
  val posIndex = spec.la.ofString(posCase)
  
  val fe = spec.fe
  val fa = fe.getAlphabet
  val laSize = spec.la.getSize
  val invLa  = spec.la.getInverseMapping
  
  val featureSet : Set[String] = fa.getMapping.keySet.toSet
  
  // val evaluator = new XGBoostEvaluator(xgSettings, laSize)
  val booster = XGBoost.loadModel(new java.io.ByteArrayInputStream(spec.booster))
  
  def decode(s: String) : Seq[(Float, Int)] = {
    val factor = fe.extractFeatures(s)
    
    val dataDm = new DMatrix(Iterator(factor) map mapMMLPFactorToLabeledPoint(false))
    val res = booster.predict(dataDm,false,0)
    val res0 = res(0)
    Seq((res0(0), 1)) // hard-coded for binary here
  }    
  
  def decodePairsMPEWithScore(li: List[StringDoublePair]) : StringDoublePair = {
    val dVec : DenseVec = DenseVec.zeros(fa.getSize)
    li foreach {s =>
      val fid = fa.ofString(s.str)
      if ((fid >= 0) && (fid < fa.getSize)) dVec(fa.ofString(s.str)) = s.value.toFloat}
    val lv = DenseVec.zeros(laSize)
    // val factor = new StdMMLPFactor(-1, dVec, lv, None)
    val dataDm = new DMatrix(dVec.asArray, 1, fa.getSize)
    val res = booster.predict(dataDm, false, 0)
    val res0 = res(0)
    new StringDoublePair(posCase, res0(0))
  }
  
  def decodePairsMPEWithScore(li: java.util.List[StringDoublePair]) : StringDoublePair = {
    val scalaList : scala.collection.mutable.Buffer[StringDoublePair] = li
    decodePairsMPEWithScore(scalaList.toList)
  }
  
}
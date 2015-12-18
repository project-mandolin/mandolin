package org.mitre.mandolin.embed

import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.util.Alphabet

class SeqInstance(val ln: Int, val seq: Array[Int]) extends Serializable 

class SeqInstanceExtractor(va: Alphabet) extends FeatureExtractor[String, SeqInstance] with Serializable {
  
  def getNumberOfFeatures = va.getSize
  def getAlphabet = va
  
  def extractFeatures(s: String) : SeqInstance = {
    var i = 0; var c = 0
    val ar = s.split("[ \n\r\t]+")
    val lb = collection.mutable.ArrayBuffer[Int]()
    while (i < ar.length) {
      val ind = va.ofString(ar(i))
      if (ind >= 0) {
        lb append ind
        c += 1
      }
      i += 1
    }    
    new SeqInstance(c, lb.toArray)
  }
  
}
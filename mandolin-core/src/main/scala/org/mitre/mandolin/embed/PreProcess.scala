package org.mitre.mandolin.embed

import collection.mutable.{HashMap, ArrayBuffer}
import org.mitre.mandolin.util.StdAlphabet

/**
 * Provides a utility routine to read in a corpus of text and compute
 * 1) a mapping from vocabulary items to integer IDs
 * 2) a set of context CBOW training instances
 * @author wellner
 */
class PreProcess(minCnt: Int) { 
  
  val unigramTableSize : Int = 1000000
  val smoothFactor : Double = 0.75
  
  val expTableSize = 1000
  
  /**
   * @param mxSize The max exponent value/magnitude
   * @return Array of size `expTableSize` that approximates e / (1.0 + e)
   */
  def constructLogisticTable(mxVal: Float) : Array[Float] = {
    val ar = Array.tabulate(1000){i =>
      val e = math.exp((i.toDouble / 1000 * 2.0 - 1.0) * mxVal)
      (e / (1.0 + e)).toFloat     
      }
    ar
  }
  
  def computeFrequencyTable(hist: HashMap[String,Int], mapping: StdAlphabet) : Array[Int] = {
    val ut = Array.fill(unigramTableSize)(0)
    val ft = new collection.mutable.HashMap[Int, Int]()
    var ss = 0 
    hist foreach {case (k,v) => 
      val id = mapping.ofString(k)
      if (id >= 0) {ft.put(mapping.ofString(k), v); ss += 1 } }    
    val ftSize = ss
    var i = 0
    var total = 0.0
    var a = 0; while (a < ftSize) { total += math.pow(ft(a),smoothFactor) ;a += 1}
    var d1 = math.pow(ft(0), smoothFactor) / total
    a = 0; while (a < unigramTableSize) {
      ut(a) = i
      if ((a.toDouble / unigramTableSize) > d1) {
        i += 1
        if (i < ftSize) d1 += math.pow(ft(i),smoothFactor) / total 
      }
      if (i >= ftSize) {
        i = ftSize - 1
      }
      a += 1
    }
    ut
  }

  def getMappingAndFreqs(dirOrFile: java.io.File, mxVal: Float = 6.0f) = {
    val hist = new HashMap[String,Int]
    if (dirOrFile.isDirectory()) dirOrFile.listFiles() foreach {f => updateTermHistogram(f,hist)}
    else updateTermHistogram(dirOrFile,hist)
    val mapping = new StdAlphabet
    hist foreach {case (s,c) => if (c >= minCnt) mapping.ofString(s) }
    mapping.ensureFixed
    val ft = computeFrequencyTable(hist, mapping)    
    (mapping, ft, constructLogisticTable(mxVal))
    //(mapping, ft)
  }
  
  val numRe = "^[0-9.,]+$".r
  
  def getNormalizedString(t: String) : String = {
    val s = t.replaceAll("[^0-9a-zA-z]", "")
    if (numRe.findFirstIn(s).isDefined) "-NUM-" else s
  }
  
  def updateTermHistogram(f: java.io.File, hist: HashMap[String,Int]) : Unit = {
    val src = io.Source.fromFile(f)("UTF-8")
    var scnt = 0
    src.getLines() foreach {l =>
      val toks = l.split("[\t\n\r ]+")
      toks foreach {t =>
        val ss = getNormalizedString(t)
        if (ss.length > 0) {
          val cnt = hist.get(ss).getOrElse(0)
          hist.put(ss,cnt+1)
        }
      }
      scnt += 1
    }
    //val histCnt = hist.get("</s>").getOrElse(0)
    //hist.put("</s>",histCnt + scnt)
  }
  
  def updateFromFile(f: java.io.File, hist: HashMap[String,Int], mapping: StdAlphabet) = {
    val src = io.Source.fromFile(f)("UTF-8")
    val sind = mapping.ofString("</s>")
    src.getLines() foreach {l =>
      val tbuf = new ArrayBuffer[Int]()
      val toks = l.split("[\t\n\r ]+")
      toks foreach {t =>
        val ss = getNormalizedString(t)        
        if (ss.length > 0) {
          val ind = mapping.ofString(ss)
          if (ind >= 0) tbuf append ind
        }
      }
    }
    mapping.ensureFixed
    mapping
  }  
}
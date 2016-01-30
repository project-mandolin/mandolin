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
  
  def computeFrequencyTable(hist: HashMap[String,Int], mapping: StdAlphabet, 
      sample: Double = 0.0, numWords: Int = 1000000) : (Array[Int], Array[Int]) = {
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
    mapping.ensureFixed
    println("mapping size = " + mapping.getSize)
    println("hist size = " + hist.size)
    // now build array with word index => probability of discarding
    val discardChances = Array.fill(ft.size)(0)
    ft foreach {case (ind,cnt) => 
      val prob = (math.sqrt(cnt.toDouble / (sample * numWords)) + 1.0) * (sample * numWords) / cnt
      discardChances(ind) = (prob * Integer.MAX_VALUE).toInt
      }
    (ut, discardChances)
  }

  def getMappingAndFreqs(dirOrFile: java.io.File, downSample: Double = 0.0) = {
    val hist = new HashMap[String,Int]
    var tw = 0
    if (dirOrFile.isDirectory()) dirOrFile.listFiles() foreach {f => tw += updateTermHistogram(f,hist)}
    else tw += updateTermHistogram(dirOrFile,hist)
    val mapping = new StdAlphabet
    hist foreach {case (s,c) => if (c >= minCnt) mapping.ofString(s) }
    mapping.ensureFixed
    val (ft,sampleTable) = computeFrequencyTable(hist, mapping, downSample, tw)    
    (mapping, ft, constructLogisticTable(6.0f), sampleTable)
  }
  
  val numRe = "^[0-9.,]+$".r
  
  def getNormalizedString(t: String) : String = {
    val s = t.replaceAll("[^0-9a-zA-z]", "")
    if (numRe.findFirstIn(s).isDefined) "-NUM-" else s
  }
  
  def updateTermHistogram(f: java.io.File, hist: HashMap[String,Int]) : Int = {
    val src = io.Source.fromFile(f)("UTF-8")
    var totalWords = 0
    src.getLines() foreach {l =>
      val toks = l.split("[\t\n\r ]+")
      toks foreach {t =>
        val ss = getNormalizedString(t)
        if (ss.length > 0) {
          totalWords += 1
          val cnt = hist.get(ss).getOrElse(0)
          hist.put(ss,cnt+1)
        }
      }
    }
    totalWords
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
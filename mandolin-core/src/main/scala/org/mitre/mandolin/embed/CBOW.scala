package org.mitre.mandolin.embed

import org.mitre.mandolin.optimize.TrainingUnitEvaluator

import xerial.larray._

/**
 * @author wellner
 */
class CBOWEvaluator(hlSize: Int, vocabSize: Int, val wSize: Int, val negSampleSize: Int, 
    freqTable: Array[Int], logisticTable: Array[Float], chances: Array[Int])
extends TrainingUnitEvaluator [SeqInstance, EmbedWeights, EmbedGradient, EmbedUpdater] with Serializable {
  
  val maxSentLength = 1000
  val ftLen = freqTable.length
  // val hlSize = emb.embW.getDim2
  // val vocabSize = emb.embW.getDim1
  val h = Array.fill(hlSize)(0.0f)
  val d = Array.fill(hlSize)(0.0f)
  val maxDp = 6.0f
  val logisticTableSizeCoef = logisticTable.length.toFloat / maxDp / 2.0f
  val eDp = 5.999f
  
  // xorshift generator with period 2^128 - 1; on the order of 10 times faster than util.Random.nextInt
  var seed0 = scala.util.Random.nextInt(Integer.MAX_VALUE)
  var seed1 = scala.util.Random.nextInt(Integer.MAX_VALUE)
  
  @inline
  private final def nextInt(m : Int) : Int = {
    var s1 = seed0
    val s0 = seed1
    seed0 = s0
    s1 ^= s1 << 23
    seed1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5)
    ((seed1 + s0) >>> 1) % m
  }
  
  
  @inline
  //private final def set(a: LArray[Float], v: Float) = { var i = 0; while (i < hlSize) { a(i) = v; i += 1} }
  private final def set(a: Array[Float], v: Float) = { var i = 0; while (i < hlSize) { a(i) = v; i += 1} }
  @inline
  //private final def timesEq(a: LArray[Float], v: Float) = { var i = 0; while (i < hlSize) { a(i) *= v; i += 1} }
  private final def timesEq(a: Array[Float], v: Float) = { var i = 0; while (i < hlSize) { a(i) *= v; i += 1} }
  @inline
  private final def logisticFn(x: Float) = logisticTable(((x + maxDp) * logisticTableSizeCoef).toInt) // 1.0 / (1.0 + math.exp(-x)) 
  
  val sent = Array.fill(maxSentLength)(0)
  
  //workhorse function that updates weights directly without returning explicit gradient
  def trainWeightsOnSequence(in: SeqInstance, w: EmbedWeights, up: EmbedUpdater) : Unit = {
    var spos = 0
    val seq = in.seq
    var bagSize = 0
    
    var con_i = 0
    // val l1Ar = w.embW.asArray // array layout of matrix
    // val l2Ar = w.outW.asArray // array layout of matrix
    
    var ii = 0
    while ((con_i < in.ln) && (ii < maxSentLength)) {      
      val ni = nextInt(Integer.MAX_VALUE)
      val wi = seq(con_i)
      val wiProb = chances(wi)
      if (ni > wiProb) {
        sent(ii) = wi
        ii += 1
      }
      con_i += 1
    }
    
    if (in.ln > 2) {
    while (spos < ii) {
      bagSize = 0
      up.updateNumProcessed()
      //val learningRate = initialLearnRate.toFloat / (1.0f + initialLearnRate.toFloat * up.totalProcessed * decay.toFloat)
      //if ((up.totalProcessed % 10000) == 0) printf("\r%d instances procssed .. learning rate %f ", up.totalProcessed, learningRate)    
      val curWord = sent(spos)
      set(h,0.0f)
      set(d,0.0f)
      val b = nextInt(wSize)
      // forward hidden outputs
      var a = b; while (a < wSize * 2 + 1 - b) {
    		con_i = spos - wSize + a
    		if ((a != wSize) && (con_i >= 0) && (con_i < in.ln) && (con_i < maxSentLength)) {
    		  val wi = sent(con_i)
    		  if (wi >= 0) {
    			  bagSize += 1
            val wiRow = w.embW.getRow(wi)
    			  var i = 0; while (i < hlSize) { h(i) += wiRow(i); i += 1}
    		  }
    		}
        a += 1
      }
      timesEq(h,(1.0f/bagSize)) // normalize h by size of input bag
      var outIndex = curWord 
      var label = 1.0f
      var s = 0; while (s < negSampleSize + 1) {
        if (s > 0) {
          val ri = nextInt(ftLen)
          outIndex = freqTable(ri)
          if (outIndex == curWord) {
            outIndex = (outIndex + nextInt(vocabSize.toInt)) % vocabSize.toInt
          }
          label = 0.0f
        }        
        // val offset = outIndex * hlSize
        var dp = 0.0f
        val outRow = w.outW.getRow(outIndex)
        var i = 0; while (i < hlSize) {
          dp += h(i) * outRow(i)         
          i += 1 } 
        val out = if (dp >= eDp) 1.0f else if (dp <= -eDp) 0.0f else logisticFn(dp)
        val o_err = (label - out)
        i = 0; while (i < hlSize) { 
          d(i) += o_err * outRow(i)
          i += 1}
        i = 0; while (i < hlSize) {
          val g = o_err * h(i)
          up.updateOutputSqG(outIndex, i, w.outW, g)
          i += 1}
        s += 1
      }
      a = b; while (a < wSize * 2 + 1 - b) {
        con_i = spos - wSize + a
        if ((a != wSize) && (con_i >= 0) && (con_i < in.ln) && (con_i < maxSentLength)) {
          //val offset = sent(con_i) * hlSize
          val wi = sent(con_i)
          var i = 0; while (i < hlSize) {
            up.updateEmbeddingSqG(wi, i, w.embW, d(i))
            //l1Ar(i + offset) += d(i)
            i += 1
          }
        }
        a += 1
      }
      spos += 1
    }
    }    
  }
      
  def evaluateTrainingUnit(unit: SeqInstance, weights: EmbedWeights, up: EmbedUpdater) : EmbedGradient = { 
    trainWeightsOnSequence(unit, weights, up)
    new EmbedGradient()
  }  
  
  def copy() = {
    // this copy will share weights but have separate layer data members
    new CBOWEvaluator(hlSize, vocabSize, wSize, negSampleSize, freqTable, logisticTable, chances) 
  }
}

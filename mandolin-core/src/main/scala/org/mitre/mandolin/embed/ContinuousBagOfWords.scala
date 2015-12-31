package org.mitre.mandolin.embed

import org.mitre.mandolin.util.{DenseTensor1 => DenseVec, LocalIOAssistant}
import org.mitre.mandolin.optimize.{TrainingUnitEvaluator}
import org.mitre.mandolin.optimize.local.{LocalOnlineOptimizer}
import org.mitre.mandolin.predict.local.LocalTrainer
import org.mitre.mandolin.config.{LearnerSettings, OnlineLearnerSettings, DecoderSettings}

class CBOWModelSettings(args: Array[String]) extends LearnerSettings(args) with OnlineLearnerSettings with DecoderSettings {
  
  val eDim        = asInt("mandolin.embed.dim")
  val contextSize = asInt("mandolin.embed.window")
  val minCnt      = asInt("mandolin.embed.min-cnt")
  val negSample   = asInt("mandolin.embed.neg-sample")

}
  

object ContinuousBagOfWords {
  
  def main(args: Array[String]) = {
    val appSettings = new CBOWModelSettings(args)
    val prep = new PreProcess(appSettings.minCnt)
    val nthreads = appSettings.numThreads
    val epochs = appSettings.numEpochs
    val inFile = appSettings.trainFile
    val eDim   = appSettings.eDim
    val (mapping, freqs, logisticTable) = prep.getMappingAndFreqs(new java.io.File(inFile.get))
    val vocabSize = mapping.getSize  
    val wts = EmbedWeights(eDim, vocabSize)     
    val fe = new SeqInstanceExtractor(mapping)
    val up = new NullUpdater
    val ev = new CBOWEvaluator(wts,appSettings.contextSize,appSettings.negSample,freqs, logisticTable, up)    
    val optimizer = new LocalOnlineOptimizer[SeqInstance, EmbedWeights, EmbedGradient, NullUpdater](wts, ev, up,epochs,1,nthreads,None)
    val trainer = new LocalTrainer(fe, optimizer)
    val io = new LocalIOAssistant
    val lines = io.readLines(inFile.get).toVector
    val (finalWeights,_) = trainer.trainWeights(lines)
    finalWeights.exportWithMapping(mapping, new java.io.File(appSettings.modelFile.get))
  }
}



/**
 * @author wellner
 */
class CBOWEvaluator(val emb: EmbedWeights, val wSize: Int, val negSampleSize: Int, freqTable: Array[Int], logisticTable: Array[Float], up: NullUpdater)
extends TrainingUnitEvaluator [SeqInstance, EmbedWeights, EmbedGradient] with Serializable {
  
  val initialLearnRate = 0.1
  val decay = 0.00001
  val ftLen = freqTable.length
  val hlSize = emb.embW.getDim2
  val vocabSize = emb.embW.getDim1
  val h = Array.fill(hlSize)(0.0f)
  val d = Array.fill(hlSize)(0.0f)
  val maxDp = 6.0f
  val logisticTableSizeCoef = logisticTable.length.toFloat / maxDp / 2.0f
  val eDp = 5.999f
  
  // xorshift generator with period 2^128 - 1
  var seed0 = util.Random.nextInt(Integer.MAX_VALUE)
  var seed1 = util.Random.nextInt(Integer.MAX_VALUE)
  
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
  private final def set(a: Array[Float], v: Float) = { var i = 0; while (i < hlSize) { a(i) = v; i += 1} }
  @inline
  private final def timesEq(a: Array[Float], v: Float) = { var i = 0; while (i < hlSize) { a(i) *= v; i += 1} }
  @inline
  private final def logisticFn(x: Float) = logisticTable(((x + maxDp) * logisticTableSizeCoef).toInt) // 1.0 / (1.0 + math.exp(-x)) 
  
  
  //workhorse function that updates weights directly without returning explicit gradient
  def trainWeightsOnSequence(in: SeqInstance, w: EmbedWeights) : Unit = {
    var spos = 0
    val seq = in.seq
    var bagSize = 0
    
    var con_i = 0
    val l1Ar = w.embW.asArray // array layout of matrix
    val l2Ar = w.outW.asArray // array layout of matrix
    
    if (in.ln > 2) {
    while (spos < in.ln) {
      bagSize = 0
      up.totalProcessed += 1
      val learningRate = initialLearnRate.toFloat / (1.0f + initialLearnRate.toFloat * up.totalProcessed * decay.toFloat)
      //if ((up.totalProcessed % 10000) == 0) printf("\r%d instances procssed .. learning rate %f ", up.totalProcessed, learningRate)    
      val curWord = seq(spos)
      set(h,0.0f)
      set(d,0.0f)
      val b = nextInt(wSize)
      // forward hidden outputs
      var a = b; while (a < wSize * 2 + 1 - b) {
    		con_i = spos - wSize + a
    		if ((a != wSize) && (con_i >= 0) && (con_i < in.ln)) {
    		  val wi = seq(con_i)
    		  if (wi >= 0) {
    			  bagSize += 1
    			  var i = 0; while (i < hlSize) { h(i) += l1Ar(i + wi * hlSize); i += 1}
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
            outIndex = (outIndex + nextInt(vocabSize)) % vocabSize
          }
          label = 0.0f
        }        
        val offset = outIndex * hlSize
        var dp = 0.0f
        var i = 0; while (i < hlSize) {
          dp += h(i) * l2Ar(offset + i)         
          i += 1 } 
        val out = if (dp >= eDp) 1.0f else if (dp <= -eDp) 0.0f else logisticFn(dp)
        val o_err = (label - out) * learningRate // XXX - AdaGrad update here         
        i = 0; while (i < hlSize) { 
          d(i) += o_err * l2Ar(offset + i)
          i += 1}
        i = 0; while (i < hlSize) { 
          l2Ar(offset + i) += o_err * h(i)
          i += 1}
        s += 1
      }
      a = b; while (a < wSize * 2 + 1 - b) {
        con_i = spos - wSize + a
        if ((a != wSize) && (con_i >= 0) && (con_i < in.ln)) {
          val wi = seq(con_i)
          var i = 0; while (i < hlSize) {             
            l1Ar(i + wi * hlSize) += d(i)
            i += 1
          }
        }
        a += 1
      }
      spos += 1
    }
    }
    
  }
      
  def evaluateTrainingUnit(unit: SeqInstance, weights: EmbedWeights) : EmbedGradient = { 
    trainWeightsOnSequence(unit, weights)
    new EmbedGradient()
  }  
  
  def copy() = {
    // this copy will share weights but have separate layer data members
    new CBOWEvaluator(emb.sharedWeightCopy(), wSize, negSampleSize, freqTable, logisticTable, up) 
  }
}

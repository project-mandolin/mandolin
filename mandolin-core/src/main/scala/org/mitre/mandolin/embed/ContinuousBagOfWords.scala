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
    val prep = new PreProcess(appSettings.contextSize)
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
class CBOWEvaluator(val emb: EmbedWeights, val wSize: Int, val negSampleSize: Int, freqTable: Array[Int], logisticTable: Array[Double], up: NullUpdater)
extends TrainingUnitEvaluator [SeqInstance, EmbedWeights, EmbedGradient] with Serializable {
  
  val initialLearnRate = 0.1
  val decay = 0.00001
  val ftLen = freqTable.length
  val hlSize = emb.embW.getDim2
  val vocabSize = emb.embW.getDim1
  val h = Array.fill(hlSize)(0.0)
  val d = Array.fill(hlSize)(0.0)
  val maxDp = 6.0
  val logisticTableSize = logisticTable.length
  
  @inline
  private final def set(a: Array[Double], v: Double) = { var i = 0; while (i < hlSize) { a(i) = v; i += 1} }
  @inline
  private final def timesEq(a: Array[Double], v: Double) = { var i = 0; while (i < hlSize) { a(i) *= v; i += 1} }
  @inline
  private final def logisticFn(x: Double) = logisticTable(((x + maxDp) * logisticTableSize / maxDp / 2.0).toInt) // 1.0 / (1.0 + math.exp(-x)) 
  
  private final def isNaN(x: Double) = !(x > 0.0) && !(x <= 0.0)
  
  //workhorse function that updates weights directly without returning explicit gradient
  def trainWeightsOnSequence(in: SeqInstance, w: EmbedWeights) : Unit = {
    var spos = 0
    val seq = in.seq
    var bagSize = 0
    val b = util.Random.nextInt(wSize)
    var con_i = 0
    val l1Ar = w.embW.asArray // array layout of matrix
    val l2Ar = w.outW.asArray // array layout of matrix
    
    if (in.ln > 2) {
    while (spos < in.ln) {
      bagSize = 0
      up.totalProcessed += 1
      val learningRate = initialLearnRate / (1.0 + initialLearnRate * up.totalProcessed * decay)
      if ((up.totalProcessed % 10000) == 0) printf("\r%d instances procssed .. learning rate %f ", up.totalProcessed, learningRate)    
      val curWord = seq(spos)
      set(h,0.0)
      set(d,0.0)
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
      timesEq(h,(1.0/bagSize)) // normalize h by size of input bag
      var outIndex = curWord 
      var label = 1.0
      var s = 0; while (s < negSampleSize + 1) {
        if (s > 0) {
          val ri = util.Random.nextInt(ftLen)
          outIndex = freqTable(ri)
          if (outIndex == curWord) {
            outIndex = (outIndex + util.Random.nextInt(vocabSize)) % vocabSize
          }
          //outIndex = (curWord + 1) % vocabSize
          label = 0.0
        }        
        val offset = outIndex * hlSize
        var dp = 0.0
        var i = 0; while (i < hlSize) {
          val cc = h(i) * l2Ar(offset + i)
          if (isNaN(cc)) println("comp is NaN with h(i) = " + h(i) + " and w=" + l2Ar(offset + i))
          dp += cc          
          i += 1 } 
        val out = if (dp > maxDp) 1.0 else if (dp < -maxDp) 0.0 else logisticFn(dp)
        val o_err = (label - out) * learningRate          
        //println("o_err = " + o_err)
        if (!(o_err > 0.0) && !(o_err <= 0.0)) {
          println("o_err = " + o_err + " with dp = " + dp)
          throw new RuntimeException("NaN reached")
        }
        i = 0; while (i < hlSize) { 
          d(i) += o_err * l2Ar(offset + i)
          //println("\tDelta to " + i + " by " + (o_err * l2Ar(offset + i)) + " resulting in " + d(i))
          i += 1}
        i = 0; while (i < hlSize) { 
          l2Ar(offset + i) += o_err * h(i)
          //println("=> Update o " + i + " by " + (o_err * h(i)) + " resulting in " + l2Ar(i + offset))
          i += 1}
        //print("o_err = " + o_err)
        //i = 0; while (i < hlSize) {print(" " + d(i)); i += 1}
        //println
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

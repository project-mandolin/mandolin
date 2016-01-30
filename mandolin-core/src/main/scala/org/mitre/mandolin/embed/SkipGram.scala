package org.mitre.mandolin.embed

import org.mitre.mandolin.util.{DenseTensor1 => DenseVec, LocalIOAssistant}
import org.mitre.mandolin.optimize.{TrainingUnitEvaluator, Updater}
import org.mitre.mandolin.optimize.local.{LocalOnlineOptimizer}
import org.mitre.mandolin.predict.local.LocalTrainer
import org.mitre.mandolin.config.{LearnerSettings, OnlineLearnerSettings, DecoderSettings}


object SkipGram {

  def main(args: Array[String]) = {
    val appSettings = new CBOWModelSettings(args)
    val prep = new PreProcess(appSettings.minCnt)
    val nthreads = appSettings.numThreads
    val epochs = appSettings.numEpochs
    val inFile = appSettings.trainFile
    val eDim = appSettings.eDim
    val downSample = appSettings.downSample
    val (mapping, freqs, logisticTable, chances) = prep.getMappingAndFreqs(new java.io.File(inFile.get), downSample)
    val vocabSize = mapping.getSize
    val wts = EmbedWeights(eDim, vocabSize)
    val fe = new SeqInstanceExtractor(mapping)
    val io = new LocalIOAssistant
    val lines = io.readLines(inFile.get).toVector
    if (appSettings.method equals "adagrad") {
      println(">> Using AdaGrad adaptive weight update scheme <<")
      val gemb = Array.fill(eDim * vocabSize)(0.0f)
      val gout = Array.fill(eDim * vocabSize)(0.0f)
      val up = new EmbedAdaGradUpdater(appSettings.initialLearnRate, gemb, gout)
      val ev = new SkipGramEvaluator[EmbedAdaGradUpdater](wts, appSettings.contextSize, appSettings.negSample, freqs, logisticTable, chances)
      val optimizer = new LocalOnlineOptimizer[SeqInstance, EmbedWeights, EmbedGradient, EmbedAdaGradUpdater](wts, ev, up, epochs, 1, nthreads, None)
      val trainer = new LocalTrainer(fe, optimizer)
      val (finalWeights, _) = trainer.trainWeights(lines)
      finalWeights.exportWithMapping(mapping, new java.io.File(appSettings.modelFile.get))
    } else {
      println(">> Using vanilla SGD weight update scheme <<")
      val up = new NullUpdater(appSettings.initialLearnRate, appSettings.sgdLambda)
      val ev = new SkipGramEvaluator[NullUpdater](wts, appSettings.contextSize, appSettings.negSample, freqs, logisticTable, chances)
      val optimizer = new LocalOnlineOptimizer[SeqInstance, EmbedWeights, EmbedGradient, NullUpdater](wts, ev, up, epochs, 1, nthreads, None)
      val trainer = new LocalTrainer(fe, optimizer)
      val (finalWeights, _) = trainer.trainWeights(lines)
      finalWeights.exportWithMapping(mapping, new java.io.File(appSettings.modelFile.get))
    }
  }
}


class SkipGramEvaluator[U <: EmbedUpdater[U]](val emb: EmbedWeights, val wSize: Int, val negSampleSize: Int, 
    freqTable: Array[Int], logisticTable: Array[Float], chances: Array[Int])
extends TrainingUnitEvaluator [SeqInstance, EmbedWeights, EmbedGradient, U] with Serializable  {

  val maxSentLength = 200
  val ftLen = freqTable.length
  val hlSize = emb.embW.getDim2
  val vocabSize = emb.embW.getDim1
  val h = Array.fill(hlSize)(0.0f)
  val d = Array.fill(hlSize)(0.0f)
  val maxDp = 6.0f
  val logisticTableSizeCoef = logisticTable.length.toFloat / maxDp / 2.0f
  val eDp = 5.999f
  
  // xorshift generator with period 2^128 - 1; on the order of 10 times faster than util.Random.nextInt
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
  def trainWeightsOnSequence(in: SeqInstance, w: EmbedWeights, up: U): Unit = {
    var spos = 0
    val seq = in.seq
    var bagSize = 0

    var con_i = 0
    val l1Ar = w.embW.asArray // array layout of matrix
    val l2Ar = w.outW.asArray // array layout of matrix
    val sent = Array.fill(maxSentLength)(0)

    var ii = 0
    while (con_i < in.ln) {
      val ni = nextInt(Integer.MAX_VALUE)
      val wi = seq(con_i)
      val wiProb = chances(wi)
      if (ni < wiProb) {
        sent(ii) = wi
        ii += 1
      }
      con_i += 1
    }
    
    if (in.ln > 2) {
      while (spos < ii) {
        up.updateNumProcessed()
        val curWord = sent(spos)
        set(h, 0.0f)
        set(d, 0.0f)
        val b = nextInt(wSize)
        var a = b; while (a < wSize * 2 + 1 - b) {
          con_i = spos - wSize + a
          if ((a != wSize) && (con_i >= 0) && (con_i < in.ln)) {
            val wi = sent(con_i)
            var i = 0; while (i < hlSize) { d(i) = 0.0f; i += 1 }            
            var inIndex = wi
            var label = 1.0f
            var s = 0;
            var outIndex = curWord
            val inOffset = inIndex * hlSize
            while (s < negSampleSize + 1) {
              if (s > 0) {
                val ri = nextInt(ftLen)
                outIndex = freqTable(ri)
                if (outIndex == curWord) {
                  outIndex = (outIndex + nextInt(vocabSize)) % vocabSize
                }
                label = 0.0f
              }
              val outOffset = outIndex * hlSize
              var dp = 0.0f
              i = 0; while (i < hlSize) {
                dp += l1Ar(inOffset + i) * l2Ar(outOffset + i)
                i += 1
              }
              val out = if (dp >= eDp) 1.0f else if (dp <= -eDp) 0.0f else logisticFn(dp)
              val o_err = (label - out)
              i = 0; while (i < hlSize) {
                d(i) += o_err * l2Ar(outOffset + i)
                i += 1
              }
              i = 0; while (i < hlSize) {
                val g = o_err * l1Ar(inOffset + i)
                up.updateOutputSqG(outOffset + i, l2Ar, g)
                i += 1
              }
              s += 1
            }
            i = 0; while (i < hlSize) {
              up.updateEmbeddingSqG(inOffset + i, l1Ar, d(i))
              i += 1
            }            
          }
          a += 1
        }
        spos += 1
      }
    }

  }
      
  def evaluateTrainingUnit(unit: SeqInstance, weights: EmbedWeights, up: U) : EmbedGradient = { 
    trainWeightsOnSequence(unit, weights, up)
    new EmbedGradient()
  }  
  
  def copy() = {
    // this copy will share weights but have separate layer data members
    new SkipGramEvaluator(emb.sharedWeightCopy(), wSize, negSampleSize, freqTable, logisticTable, chances) 
  }
}
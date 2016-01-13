package org.mitre.mandolin.embed.spark

import org.mitre.mandolin.embed.{PreProcess, EmbedWeights, SeqInstanceExtractor, NullUpdater, EmbedAdaGradUpdater,
  CBOWEvaluator, SeqInstance, EmbedGradient, EmbedUpdater}
import org.mitre.mandolin.util.spark.{SparkIOAssistant}
import org.mitre.mandolin.util.{Alphabet, StdAlphabet}
import org.mitre.mandolin.optimize.spark.DistributedOnlineOptimizer
import org.mitre.mandolin.predict.spark.Trainer
import org.mitre.mandolin.glp.spark.AppConfig


import org.apache.spark.{SparkContext, AccumulatorParam}
import org.apache.spark.rdd.RDD

object CBOWDist {
  
  val unigramTableSize = 10000000
  
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
  
  val numRe = "^[0-9.,]+$".r
  
  def getNormalizedString(t: String) : String = {
    val s = t.replaceAll("[^0-9a-zA-z]", "")
    if (numRe.findFirstIn(s).isDefined) "-NUM-" else s
  }
  
  def getVocabularyAndFreqs(lines: RDD[String], minCnt: Int, smoothFactor: Double) = {
    val _mc = minCnt
    val res = lines.flatMap { x => x.split("[ \n\r\t]+") }
               .map{x => (getNormalizedString(x),1)}.filter{x => x._1.length() > 0} 
               .reduceByKey {_ + _} filter {x => x._2 >= _mc }
    val hist = res.collect()
    val size = hist.length
    val alphabet = new StdAlphabet
    val ft = Array.tabulate(size){i => alphabet.ofString(hist(i)._1); hist(i)._2}
    alphabet.ensureFixed
    val ut = Array.fill(unigramTableSize)(0)
    var i = 0
    var total = 0.0
    var a = 0; while (a < size) { total += math.pow(ft(a),smoothFactor) ;a += 1}
    var d1 = math.pow(ft(0), smoothFactor) / total    
    a = 0; while (a < unigramTableSize) {
      ut(a) = i
      if ((a.toDouble / unigramTableSize) > d1) {
        i += 1
        if (i < size) d1 += math.pow(ft(i),smoothFactor) / total 
      }
      if (i >= size) {
        i = size - 1
      }
      a += 1
    }
    (alphabet, ut, constructLogisticTable(6.0f))
  }  

  def main(args: Array[String]) = {
    val appSettings = new org.mitre.mandolin.embed.CBOWModelSettings(args)
    val nthreads = appSettings.numThreads
    val epochs = appSettings.numEpochs
    val inFile = appSettings.trainFile
    val eDim   = appSettings.eDim
    val sc = AppConfig.getSparkContext(appSettings)
    val lines = sc.textFile(inFile.get)
    val (mapping, freqs, logisticTable) = getVocabularyAndFreqs(lines, appSettings.minCnt, 0.75)
    val vocabSize = mapping.getSize  
    val wts = EmbedWeights(eDim, vocabSize) 
    val fe = new SeqInstanceExtractor(mapping)
    val gemb = Array.fill(eDim * vocabSize)(0.0f)
    val gout = Array.fill(eDim * vocabSize)(0.0f)
    println("*** Number of parameters = " + (eDim * vocabSize * 2))
    val lines1 = lines.repartition(appSettings.numPartitions * appSettings.numThreads).coalesce(appSettings.numPartitions, false) // repartition should balance these across cluster ..
    if (appSettings.method equals "adagrad") {
      println("*** Using AdaGrad stochastic optimization ***")
      val up = new EmbedAdaGradUpdater(appSettings.initialLearnRate, gemb, gout)            
      val ev = new CBOWEvaluator[EmbedAdaGradUpdater](wts,appSettings.contextSize,appSettings.negSample, freqs, logisticTable)
      val optimizer = 
        new DistributedOnlineOptimizer[SeqInstance, EmbedWeights, EmbedGradient, EmbedAdaGradUpdater](sc, wts, ev, up, epochs,1,nthreads,None)
      val trainer = new Trainer(fe, optimizer)
      //val io = new SparkIOAssistant(sc)
      val (finalWeights,_) = trainer.trainWeights(lines1)
      finalWeights.exportWithMapping(mapping, new java.io.File(appSettings.modelFile.get))
    }
    else {
      println("*** Using SGD optimization ***")
      val up = new NullUpdater(appSettings.initialLearnRate, appSettings.sgdLambda)            
      val ev = new CBOWEvaluator[NullUpdater](wts,appSettings.contextSize,appSettings.negSample,freqs, logisticTable)      
      val optimizer = new DistributedOnlineOptimizer[SeqInstance, EmbedWeights, EmbedGradient, NullUpdater](sc, wts, ev, up, epochs,1,nthreads,None)
      val trainer = new Trainer(fe, optimizer)
      val io = new SparkIOAssistant(sc)
      val (finalWeights,_) = trainer.trainWeights(lines1)
      finalWeights.exportWithMapping(mapping, new java.io.File(appSettings.modelFile.get))
    }
  }

}
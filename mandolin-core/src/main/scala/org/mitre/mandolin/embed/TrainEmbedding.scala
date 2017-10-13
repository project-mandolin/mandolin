package org.mitre.mandolin.embed

import org.mitre.mandolin.util.LocalIOAssistant
import org.mitre.mandolin.optimize.standalone.{OnlineOptimizer}
import org.mitre.mandolin.predict.standalone.Trainer
import org.mitre.mandolin.config.ConfigGeneratedCommandOptions
import org.mitre.mandolin.mlp.MandolinMLPSettings
import org.mitre.mandolin.util.{DenseTensor2, Simple2DArray}
import com.typesafe.config.Config
import xerial.larray._

class EmbeddingModelSettings(args: Array[String]) extends MandolinMLPSettings(args) {
 
  val embedMethod = asStr("mandolin.embed.method")
  val eDim        = asInt("mandolin.embed.dim")
  val contextSize = asInt("mandolin.embed.window")
  val minCnt      = asInt("mandolin.embed.min-cnt")
  val negSample   = asInt("mandolin.embed.neg-sample")
  val downSample  = asFloatOpt("mandolin.embed.down-sample").getOrElse(1E-5f).toDouble
}


object TrainEmbedding {
  
  def main(args: Array[String]) : Unit = {
    val appSettings = new EmbeddingModelSettings(args)
    val prep = new PreProcess(appSettings.minCnt)
    val nthreads = appSettings.numThreads
    val epochs = appSettings.numEpochs
    val inFile = appSettings.trainFile
    val eDim   = appSettings.eDim
    val downSample = appSettings.downSample
    val (mapping, freqs, logisticTable,chances) = prep.getMappingAndFreqs(new java.io.File(inFile.get), downSample)
    val vocabSize = mapping.getSize  
    val wts = EmbedWeights(eDim, vocabSize)     
    val fe = new SeqInstanceExtractor(mapping)
    val io = new LocalIOAssistant
    val lines = io.readLines(inFile.get).toVector
    
    if (appSettings.method equals "adagrad") {
      println(">> Using AdaGrad adaptive weight update scheme <<")
      // XXX - redo the updaters to use arrays of arrays
      // val gemb = Array.fill(eDim.toLong * vocabSize.toLong)
      // val gemb = Array.fill(eDim * vocabSize)(0.0f)
      // val gout = Array.fill(eDim * vocabSize)(0.0f)
      val gemb = Simple2DArray.floatArray(vocabSize, eDim)
      val gout = Simple2DArray.floatArray(vocabSize, eDim)
      val up = new EmbedAdaGradUpdater(appSettings.initialLearnRate, gemb, gout)
      val ev = 
      if (appSettings.embedMethod.equals("skipgram")) 
        new SkipGramEvaluator[EmbedAdaGradUpdater](wts,appSettings.contextSize,appSettings.negSample,freqs, logisticTable, chances)
      else
        new CBOWEvaluator[EmbedAdaGradUpdater](wts,appSettings.contextSize,appSettings.negSample,freqs, logisticTable, chances)      
      val optimizer = new OnlineOptimizer[SeqInstance, EmbedWeights, EmbedGradient, EmbedAdaGradUpdater](wts, ev, up,epochs,1,nthreads,None)
      val trainer = new Trainer(fe, optimizer)
      val (finalWeights,_) = trainer.trainWeights(lines)
      finalWeights.exportWithMapping(mapping, new java.io.File(appSettings.modelFile.get))
    } else {
      println(">> Using vanilla SGD weight update scheme <<")
      val up = new NullUpdater(appSettings.initialLearnRate, appSettings.sgdLambda)
      val ev =
        if (appSettings.embedMethod.equals("skipgram"))
          new SkipGramEvaluator[NullUpdater](wts,appSettings.contextSize,appSettings.negSample,freqs, logisticTable, chances)
        else 
          new CBOWEvaluator[NullUpdater](wts,appSettings.contextSize,appSettings.negSample,freqs, logisticTable, chances)          
      val optimizer = new OnlineOptimizer[SeqInstance, EmbedWeights, EmbedGradient, NullUpdater](wts, ev, up,epochs,1,nthreads,None)
      val trainer = new Trainer(fe, optimizer)
      val (finalWeights,_) = trainer.trainWeights(lines)
      finalWeights.exportWithMapping(mapping, new java.io.File(appSettings.modelFile.get))
    }
  } 

}
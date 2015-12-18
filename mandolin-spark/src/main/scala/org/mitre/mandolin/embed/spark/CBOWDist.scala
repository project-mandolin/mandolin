package org.mitre.mandolin.embed.spark

import org.mitre.mandolin.embed.{PreProcess, EmbedWeights, SeqInstanceExtractor, NullUpdater, CBOWEvaluator, SeqInstance, EmbedGradient}
import org.mitre.mandolin.util.spark.SparkIOAssistant
import org.mitre.mandolin.optimize.spark.DistributedOnlineOptimizer
import org.mitre.mandolin.predict.spark.Trainer

import org.apache.spark.SparkContext

class CBOWDist {

  def main(args: Array[String]) = {
    val prep = new PreProcess(1)
    val nthreads = if (args.length > 3) args(3).toInt else 8
    val inFile = args(0)
    val eDim   = args(1).toInt
    val (mapping, freqs) = prep.getMappingAndFreqs(new java.io.File(inFile))
    val vocabSize = mapping.getSize  
    val wts = EmbedWeights(eDim, vocabSize)     
    val fe = new SeqInstanceExtractor(mapping)
    val up = new NullUpdater
    val ev = new CBOWEvaluator(wts,5,10,freqs, up)
    val sc = new SparkContext
    val optimizer = new DistributedOnlineOptimizer[SeqInstance, EmbedWeights, EmbedGradient, NullUpdater](sc, wts, ev, up,10,1,nthreads,None)
    val trainer = new Trainer(fe, optimizer)
    val io = new SparkIOAssistant(sc)
    val lines = sc.textFile(inFile, 10)
    val (finalWeights,_) = trainer.trainWeights(lines)
    finalWeights.exportWithMapping(mapping, new java.io.File(args(2)))
  }

}
package org.mitre.mandolin.mx.standalone

import org.mitre.mandolin.mlp.{ AbstractProcessor, MandolinMLPSettings }
import org.mitre.mandolin.util.{ StdAlphabet, RandomAlphabet, Alphabet, IOAssistant, LocalIOAssistant }
import org.mitre.mandolin.predict.standalone.Trainer
import org.slf4j.LoggerFactory

class StandaloneMxNetProcessor(appSettings: MandolinMLPSettings) extends AbstractProcessor {
  private val logger = LoggerFactory.getLogger(this.getClass)
  val io = new LocalIOAssistant
  def getFeatureExtractor = getComponentsViaSettings(appSettings, io).featureExtractor
  
  def processTrain() = {
    val components = getComponentsViaSettings(appSettings, io)       
    val trainFile = appSettings.trainFile
    val fe = getFeatureExtractor
    val optimizer = StandaloneMxNetOptimizer.getStandaloneOptimizer()
    logger.info(".. reading data ..")
    val lines = io.readLines(trainFile.get)
    val trainer = new Trainer(fe, optimizer)
    val (finalWeights,_) = trainer.trainWeights(lines.toVector)    
    finalWeights
  }
}

object RunStandaloneMxNetProcessor {
  
  def main(args: Array[String]) : Unit = {
    val appSettings = new MandolinMLPSettings(args)
    (new StandaloneMxNetProcessor(appSettings)).processTrain()
  }
}
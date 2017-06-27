package org.mitre.mandolin.mx.local

import org.mitre.mandolin.mlp.{ AbstractProcessor, MandolinMLPSettings }
import org.mitre.mandolin.util.{ StdAlphabet, RandomAlphabet, Alphabet, IOAssistant, LocalIOAssistant }
import org.mitre.mandolin.predict.standalone.Trainer
import org.slf4j.LoggerFactory

class LocalMxNetProcessor(appSettings: MandolinMLPSettings) extends AbstractProcessor {
  private val logger = LoggerFactory.getLogger(classOf[LocalMxNetProcessor])
  val io = new LocalIOAssistant
  def getFeatureExtractor = getComponentsViaSettings(appSettings, io).featureExtractor
  
  def processTrain() = {
    val components = getComponentsViaSettings(appSettings, io)       
    val trainFile = appSettings.trainFile
    val fe = getFeatureExtractor
    val optimizer = LocalMxNetOptimizer.getLocalOptimizer()
    logger.info(".. reading data ..")
    val lines = io.readLines(trainFile.get)
    val trainer = new Trainer(fe, optimizer)
    val (finalWeights,_) = trainer.trainWeights(lines.toVector)    
    finalWeights
  }
}

object RunLocalMxNetProcessor {
  
  def main(args: Array[String]) : Unit = {
    val appSettings = new MandolinMLPSettings(args)
    (new LocalMxNetProcessor(appSettings)).processTrain()
  }
}
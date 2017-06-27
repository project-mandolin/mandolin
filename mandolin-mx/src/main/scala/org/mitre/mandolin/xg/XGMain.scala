package org.mitre.mandolin.xg

import org.mitre.mandolin.mlp.{ MMLPFactor, MMLPTrainerBuilder }
import org.mitre.mandolin.util.LocalIOAssistant

object XGMain extends org.mitre.mandolin.config.LogInit {
  
  def getVecIOs(appSettings: XGModelSettings) : (Vector[MMLPFactor], Option[Vector[MMLPFactor]], Int) = {
    val io = new LocalIOAssistant
    val components = MMLPTrainerBuilder.getComponentsViaSettings(appSettings, io)
    val featureExtractor = components.featureExtractor
    val trFile = appSettings.trainFile.get
    val tstFile = appSettings.testFile
    val trVecs = (io.readLines(trFile) map { l => featureExtractor.extractFeatures(l) } toVector)
    val tstVecs = tstFile map {tf => (io.readLines(tf) map { l => featureExtractor.extractFeatures(l) } toVector) }
    (trVecs, tstVecs, featureExtractor.getNumberOfFeatures)
  }
  
  def main(args: Array[String]) : Unit = {
    val xgSettings = new XGModelSettings(args)
    val (trVecs, tstVecs, _) = getVecIOs(xgSettings)
    val evaluator = new XGBoostEvaluator(xgSettings)
    val finalMetric = evaluator.evaluateTrainingSet(trVecs.toIterator, tstVecs map { _.toIterator })
    
  }

}
package org.mitre.mandolin.xg

import org.mitre.mandolin.glp.{ GLPFactor, GLPTrainerBuilder }
import org.mitre.mandolin.util.LocalIOAssistant

object XGMain extends org.mitre.mandolin.config.LogInit with org.mitre.mandolin.app.AppMain {
  
  def getVecIOs(appSettings: XGModelSettings) : (Vector[GLPFactor], Option[Vector[GLPFactor]], Int, Int) = {
    val io = new LocalIOAssistant
    val components = GLPTrainerBuilder.getComponentsViaSettings(appSettings, io)
    val featureExtractor = components.featureExtractor
    val trFile = appSettings.trainFile.get
    val tstFile = appSettings.testFile
    val trVecs = (io.readLines(trFile) map { l => featureExtractor.extractFeatures(l) } toVector)
    val tstVecs = tstFile map {tf => (io.readLines(tf) map { l => featureExtractor.extractFeatures(l) } toVector) }
    (trVecs, tstVecs, featureExtractor.getNumberOfFeatures, components.labelAlphabet.getSize)
  }
  
  def main(args: Array[String]) : Unit = {
    val xgSettings = new XGModelSettings(args)
    val (trVecs, tstVecs, _, numLabels) = getVecIOs(xgSettings)
    val evaluator = new XGBoostEvaluator(xgSettings, numLabels)
    val (finalMetric, booster) = evaluator.evaluateTrainingSet(trVecs.toIterator, tstVecs map { _.toIterator })
    booster foreach {b =>
      xgSettings.modelFile match {
        case Some(m) => b.saveModel(xgSettings.modelFile.get)
        case None => throw new RuntimeException("Destination model file path expected")
      }
    }
    println("Training complete.")
    println("Final evaluation metric result: " + finalMetric)
  }

}
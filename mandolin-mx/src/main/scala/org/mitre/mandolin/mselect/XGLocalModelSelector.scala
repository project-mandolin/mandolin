package org.mitre.mandolin.mselect

import org.mitre.mandolin.app.AppMain
import org.mitre.mandolin.xg.XGModelSettings
import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.mlp.{MMLPFactor, MMLPTrainerBuilder, MandolinMLPSettings}
import org.mitre.mandolin.util.LocalIOAssistant

class LocalXGModelSelector(val msb: XGModelSpaceBuilder, trainFile: String, testFile: Option[String], numWorkers: Int, 
    scoreSampleSize: Int, acqFunRelearnSize: Int, totalEvals: Int,
    appSettings: Option[XGModelSettings with ModelSelectionSettings] = None, useHyperband: Boolean = false, hyperMix: Float = 1.0f,
    hyperMax: Int = 81) 
extends ModelSelectionDriver(trainFile, testFile, numWorkers, scoreSampleSize, acqFunRelearnSize, totalEvals, useHyperband, hyperMix, hyperMax) {
  
  // allow for Mandolin to use the appSettings here while programmatic/external setup can be done directly by passing
  // in various parameters
  def this(_msb: XGModelSpaceBuilder, appSettings: XGModelSettings with ModelSelectionSettings) = { 
    this(_msb, appSettings.trainFile.get, appSettings.testFile, appSettings.numWorkers, 
    appSettings.scoreSampleSize, appSettings.updateFrequency, appSettings.totalEvals, Some(appSettings), appSettings.useHyperband,
    appSettings.hyperbandMixParam, appSettings.numEpochs)
  }
  
  val acqFun = appSettings match {case Some(s) => s.acquisitionFunction case None => new RandomAcquisition }
  
  val fe: FeatureExtractor[String, MMLPFactor] = {
    val settings = appSettings.getOrElse((new MandolinMLPSettings).withSets(Seq(
      ("mandolin.trainer.train-file", trainFile),
      ("mandolin.trainer.test-file", testFile)
    )))
    val io = new LocalIOAssistant
    val components = MMLPTrainerBuilder.getComponentsViaSettings(settings, io)
    val featureExtractor = components.featureExtractor
    featureExtractor
  }
  val ms: ModelSpace = msb.build(appSettings)
  override val ev = {
    val io = new LocalIOAssistant
    val trVecs = (io.readLines(trainFile) map { l => fe.extractFeatures(l) } toVector)
    val tstVecs = testFile map {tf =>  (io.readLines(tf) map { l => fe.extractFeatures(l) } toVector) }
    new XGModelEvaluator(trVecs, tstVecs)
  }
}

/**
 * @author wellner
 */
object XGLocalModelSelector extends org.mitre.mandolin.config.LogInit with AppMain {

   def main(args: Array[String]): Unit = {
    val appSettings1 = new XGModelSettings(args) with ModelSelectionSettings
    val appSettings2 = if (appSettings1.useHyperband && appSettings1.useCheckpointing) appSettings1 else {
      appSettings1.withSets(Seq(("mandolin.trainer.model-file","null"))) // clunky - but ensure this is null if we're running model selection distributed
    }
    val appSettings = new XGModelSettings(None,Some(appSettings2.config)) with ModelSelectionSettings
    val builder1 = new XGModelSpaceBuilder(appSettings.modelSpace) // MxLearnerFactory    
    val selector = new LocalXGModelSelector(builder1, appSettings)
    selector.search()    
  }
}

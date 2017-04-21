package org.mitre.mandolin.mselect

import org.mitre.mandolin.mx.MxModelSettings
import org.mitre.mandolin.util.LocalIOAssistant
import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.glp.{ GLPModelSettings, GLPTrainerBuilder, GLPFactor }


class LocalMxModelSelector(val msb: MxModelSpaceBuilder, trainFile: String, testFile: String, numWorkers: Int, scoreSampleSize: Int, acqFunRelearnSize: Int, totalEvals: Int,
    appSettings: Option[MxModelSettings with ModelSelectionSettings] = None, useHyperband: Boolean = false) 
extends ModelSelectionDriver(trainFile, testFile, numWorkers, scoreSampleSize, acqFunRelearnSize, totalEvals, useHyperband) {
  
  // allow for Mandolin to use the appSettings here while programmatic/external setup can be done directly by passing
  // in various parameters
  def this(_msb: MxModelSpaceBuilder, appSettings: MxModelSettings with ModelSelectionSettings) = { 
    this(_msb, appSettings.trainFile.get, appSettings.testFile.getOrElse(appSettings.trainFile.get), appSettings.numWorkers, 
    appSettings.scoreSampleSize, appSettings.updateFrequency, appSettings.totalEvals, Some(appSettings), appSettings.useHyperband)
  }
  
  val acqFun = appSettings match {case Some(s) => s.acquisitionFunction case None => new RandomAcquisition }
  
  val (fe: FeatureExtractor[String, GLPFactor], numInputs: Int, numOutputs: Int) = {
    val settings = appSettings.getOrElse((new GLPModelSettings).withSets(Seq(
      ("mandolin.trainer.train-file", trainFile),
      ("mandolin.trainer.test-file", testFile),
      ("mandolin.trainer.specification", null) // force this to be null
    )))
    val io = new LocalIOAssistant
    val components = GLPTrainerBuilder.getComponentsViaSettings(settings, io)
    val featureExtractor = components.featureExtractor
    val numInputs = featureExtractor.getNumberOfFeatures
    val numOutputs = appSettings.get.numberOfClasses
    (featureExtractor, numInputs, numOutputs)
  }
  val ms: ModelSpace = msb.build(numInputs, numOutputs, false, appSettings)
  override val ev = {
    val io = new LocalIOAssistant
    val trVecs = (io.readLines(trainFile) map { l => fe.extractFeatures(l) } toVector)
    val tstVecs = (io.readLines(testFile) map { l => fe.extractFeatures(l) } toVector)
    new LocalMxModelEvaluator(trVecs, tstVecs)
  }
}

class LocalFileSystemImgMxModelSelector(val msb: MxModelSpaceBuilder, trainFile: String, testFile: String, numWorkers: Int, scoreSampleSize: Int, acqFunRelearnSize: Int, totalEvals: Int,
    appSettings: Option[MxModelSettings with ModelSelectionSettings] = None, useHyperband : Boolean = false) 
extends ModelSelectionDriver(trainFile, testFile, numWorkers, scoreSampleSize, acqFunRelearnSize, totalEvals, useHyperband) {
  
  // allow for Mandolin to use the appSettings here while programmatic/external setup can be done directly by passing
  // in various parameters
  def this(_msb: MxModelSpaceBuilder, appSettings: MxModelSettings with ModelSelectionSettings) = { 
    this(_msb, appSettings.trainFile.get, appSettings.testFile.getOrElse(appSettings.trainFile.get), appSettings.numWorkers, 
    appSettings.scoreSampleSize, appSettings.updateFrequency, appSettings.totalEvals, Some(appSettings), appSettings.useHyperband)
  }
  val acqFun = appSettings match {case Some(s) => s.acquisitionFunction case None => new RandomAcquisition }
  
  val ms: ModelSpace = msb.build(0, 0, false, appSettings)
  override val ev = {
    new FileSystemMxModelEvaluator(new java.io.File(trainFile), new java.io.File(testFile))
  }
}

object MxLocalModelSelector extends org.mitre.mandolin.config.LogInit {
   def main(args: Array[String]): Unit = {
    val appSettings = new MxModelSettings(args) with ModelSelectionSettings
    val builder1 = new MxModelSpaceBuilder(appSettings.modelSpace) // MxLearnerFactory
    if ((appSettings.inputType equals "recordio") || (appSettings.inputType equals "mnist")) {       
      val selector = new LocalFileSystemImgMxModelSelector(builder1, appSettings)
      selector.search()
    } else {
      val selector = new LocalMxModelSelector(builder1, appSettings)
      selector.search()
    }    
  }
}


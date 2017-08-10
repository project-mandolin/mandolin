package org.mitre.mandolin.mselect

import org.mitre.mandolin.mx.MxModelSettings
import org.apache.spark.SparkContext
import org.mitre.mandolin.app.AppMain
import org.mitre.mandolin.util.LocalIOAssistant
import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.mlp.{ANNetwork, CategoricalMMLPPredictor, MMLPFactor, MMLPTrainerBuilder, MMLPWeights, MandolinMLPSettings, SparseInputLType}

class SparkMxModelSelectionDriver(val sc: SparkContext, val msb: MxModelSpaceBuilder, trainFile: String, testFile: Option[String],
    numWorkers: Int, scoreSampleSize: Int, acqFunRelearnSize: Int, totalEvals: Int,
    appSettings: Option[MxModelSettings with ModelSelectionSettings] = None, useHyperband: Boolean = false, hyperMix: Float = 1.0f,
    hyperMax: Int = 81)
extends ModelSelectionDriver(trainFile, testFile, numWorkers, scoreSampleSize, acqFunRelearnSize, totalEvals, useHyperband, hyperMix,hyperMax) {

  def this(sc: SparkContext, _msb: MxModelSpaceBuilder, appSettings: MxModelSettings with ModelSelectionSettings) = {
    this(sc, _msb, appSettings.trainFile.get, appSettings.testFile, appSettings.numWorkers,
    appSettings.scoreSampleSize, appSettings.updateFrequency, appSettings.totalEvals, Some(appSettings), appSettings.useHyperband,
    appSettings.hyperbandMixParam, appSettings.numEpochs)
  }
  val acqFun = appSettings match {case Some(s) => s.acquisitionFunction case None => new RandomAcquisition }

  val (fe: FeatureExtractor[String, MMLPFactor], numInputs: Int, numOutputs: Int) = {
    val settings = appSettings.getOrElse((new MandolinMLPSettings).withSets(Seq(
      ("mandolin.trainer.train-file", trainFile),
      ("mandolin.trainer.test-file", testFile)
    )))
    val io = new LocalIOAssistant
    val components = MMLPTrainerBuilder.getComponentsViaSettings(settings, io)
    val featureExtractor = components.featureExtractor
    val numInputs = featureExtractor.getNumberOfFeatures
    val numOutputs = appSettings.get.numberOfClasses
    (featureExtractor, numInputs, numOutputs)
  }
  val ms: ModelSpace = msb.build(numInputs, numOutputs, false, appSettings)
  override val ev = {
    val io = new LocalIOAssistant
    val trVecs = (io.readLines(trainFile) map { l => fe.extractFeatures(l) } toVector)
    val tstVecs = testFile match {case Some(tst) => (io.readLines(tst) map { l => fe.extractFeatures(l) } toVector) case None => trVecs }
    new SparkMxModelEvaluator(sc, sc.broadcast(trVecs), sc.broadcast(tstVecs))
  }
}

class SparkLocalFileSystemImgMxModelSelector(val sc: SparkContext, val msb: MxModelSpaceBuilder, trainFile: String, testFile: Option[String], 
    numWorkers: Int, scoreSampleSize: Int, acqFunRelearnSize: Int, totalEvals: Int,
    appSettings: Option[MxModelSettings with ModelSelectionSettings] = None, useHyperband: Boolean = false, hyperMix: Float = 1.0f,
    hyperMax: Int = 81)
extends ModelSelectionDriver(trainFile, testFile, numWorkers, scoreSampleSize, acqFunRelearnSize, totalEvals, useHyperband, hyperMix, hyperMax) {


  // allow for MandolinMain to use the appSettings here while programmatic/external setup can be done directly by passing
  // in various parameters
  def this(sc: SparkContext, _msb: MxModelSpaceBuilder, appSettings: MxModelSettings with ModelSelectionSettings) = {
    this(sc, _msb, appSettings.trainFile.get, appSettings.testFile, appSettings.numWorkers,
    appSettings.scoreSampleSize, appSettings.updateFrequency, appSettings.totalEvals, Some(appSettings), appSettings.useHyperband,
    appSettings.hyperbandMixParam, appSettings.numEpochs)
  }
  val acqFun = appSettings match {case Some(s) => s.acquisitionFunction case None => new RandomAcquisition }

  val ms: ModelSpace = msb.build(0, 0, false, appSettings)
  override val ev = {
    new SparkMxFileSystemModelEvaluator(sc, trainFile, testFile.getOrElse(trainFile))
  }
}

object SparkMxModelSelectionDriver extends AppMain {

  def main(args: Array[String]) : Unit = {
    val a1 = new MxModelSettings(args)
    val appSettings1 = new MxModelSettings(None,Some(a1.config)) with ModelSelectionSettings
    val appSettings2 = if (appSettings1.useHyperband && appSettings1.useCheckpointing) appSettings1 else {
      appSettings1.withSets(Seq(("mandolin.trainer.model-file","null"))) // clunky - but ensure this is null if we're running model selection distributed
    }
    val appSettings = new MxModelSettings(None,Some(appSettings2.config)) with ModelSelectionSettings
    val sc = new SparkContext
    val trainFile = appSettings.trainFile.get
    val testFile = appSettings.testFile.getOrElse(trainFile)
    val builder = new MxModelSpaceBuilder(appSettings.modelSpace)
    val selector =
      if ((appSettings.inputType equals "recordio") || (appSettings.inputType equals "mnist"))
        new SparkLocalFileSystemImgMxModelSelector(sc, builder, appSettings)
      else new SparkMxModelSelectionDriver(sc, builder, appSettings)
    selector.search()
  }  

}
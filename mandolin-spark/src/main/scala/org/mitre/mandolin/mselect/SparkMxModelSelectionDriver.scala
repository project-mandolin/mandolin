package org.mitre.mandolin.mselect

import org.mitre.mandolin.mx.MxModelSettings
import org.apache.spark.SparkContext

import org.mitre.mandolin.util.LocalIOAssistant
import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.glp.{ GLPTrainerBuilder, GLPModelSettings, CategoricalGLPPredictor, GLPFactor, GLPWeights, ANNetwork, SparseInputLType }

class SparkMxModelSelectionDriver(val sc: SparkContext, val msb: MxModelSpaceBuilder, trainFile: String, testFile: String, 
    numWorkers: Int, workerBatchSize: Int, scoreSampleSize: Int, acqFunRelearnSize: Int, totalEvals: Int,
    appSettings: Option[MxModelSettings with ModelSelectionSettings] = None) 
extends ModelSelectionDriver(trainFile, testFile, numWorkers, workerBatchSize, scoreSampleSize, acqFunRelearnSize, totalEvals) {
  
  def this(sc: SparkContext, _msb: MxModelSpaceBuilder, appSettings: MxModelSettings with ModelSelectionSettings) = { 
    this(sc, _msb, appSettings.trainFile.get, appSettings.testFile.getOrElse(appSettings.trainFile.get), appSettings.numWorkers, appSettings.workerBatchSize, 
    appSettings.scoreSampleSize, appSettings.updateFrequency, appSettings.totalEvals, Some(appSettings))
  }      
  
  val (fe: FeatureExtractor[String, GLPFactor], numInputs: Int, numOutputs: Int) = {
    val settings = appSettings.getOrElse((new GLPModelSettings).withSets(Seq(
      ("mandolin.trainer.train-file", trainFile),
      ("mandolin.trainer.test-file", testFile)
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
    new SparkMxModelEvaluator(sc, sc.broadcast(trVecs), sc.broadcast(tstVecs))
  }
}

object SparkMxModelSelectionDriver extends org.mitre.mandolin.config.LogInit {

  def main(args: Array[String]) : Unit = {
    val appSettings = new MxModelSettings(args) with ModelSelectionSettings
    val sc = new SparkContext
    val trainFile = appSettings.trainFile.get
    val testFile = appSettings.testFile.getOrElse(trainFile)
    val builder = MxLearnerFactory.getModelSpaceBuilder(appSettings.modelSpace)    
    val selector = new SparkMxModelSelectionDriver(sc, builder, appSettings)
    selector.search()
  }
  

}
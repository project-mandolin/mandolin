package org.mitre.mandolin.mselect

import org.mitre.mandolin.mselect.WorkPullingPattern.RegisterWorker
import org.mitre.mandolin.util.LocalIOAssistant
import akka.actor.{PoisonPill, ActorSystem, Props}
import scala.concurrent.{ExecutionContext }
import java.util.concurrent.Executors
import org.mitre.mandolin.glp.{ GLPTrainerBuilder, GLPModelSettings, CategoricalGLPPredictor, GLPFactor, GLPWeights, ANNetwork, SparseInputLType }
import org.mitre.mandolin.predict.local.NonExtractingEvalDecoder
import org.mitre.mandolin.predict.{ DiscreteConfusion }
import org.mitre.mandolin.transform.FeatureExtractor


class LocalModelSelector(val msb: MandolinModelSpaceBuilder, trainFile: String, testFile: String, numWorkers: Int, 
    scoreSampleSize: Int, acqFunRelearnSize: Int, totalEvals: Int,
    appSettings: Option[GLPModelSettings with ModelSelectionSettings] = None, useHyperband: Boolean = false, hyperMix: Float = 1.0f,
    hyperMax: Int = 81) 
extends ModelSelectionDriver(trainFile, testFile, numWorkers, scoreSampleSize, acqFunRelearnSize, totalEvals, useHyperband, hyperMix, hyperMax) {
  
  def this(_msb: MandolinModelSpaceBuilder, appSettings: GLPModelSettings with ModelSelectionSettings) = { 
    this(_msb, appSettings.trainFile.get, appSettings.testFile.getOrElse(appSettings.trainFile.get), appSettings.numWorkers, 
    appSettings.scoreSampleSize, appSettings.updateFrequency, appSettings.totalEvals, Some(appSettings), appSettings.useHyperband, 
    appSettings.hyperbandMixParam, appSettings.numEpochs)
  }
  
  val acqFun = appSettings match {case Some(s) => s.acquisitionFunction case None => new ExpectedImprovement}
  
  val (fe: FeatureExtractor[String, GLPFactor], nnet: ANNetwork, numInputs: Int, numOutputs: Int, sparse: Boolean) = {
    val settings = appSettings.getOrElse((new GLPModelSettings).withSets(Seq(
      ("mandolin.trainer.train-file", trainFile),
      ("mandolin.trainer.test-file", testFile)
    )))

    val (trainer, nn) = GLPTrainerBuilder(settings)
    val featureExtractor = trainer.getFe
    featureExtractor.getAlphabet.ensureFixed // fix the alphabet
    val numInputs = nn.inLayer.getNumberOfOutputs // these will then be gathered dynamically from the trainFile
    val numOutputs = nn.outLayer.getNumberOfOutputs // ditto
    val isSparse = nn.inLayer.ltype.designate match {case SparseInputLType => true case _ => false}
    (featureExtractor, nn, numInputs, numOutputs, isSparse)
  }
  
  val ms: ModelSpace = msb.build(numInputs, numOutputs, sparse, appSettings)
  
  val ev = {
    val io = new LocalIOAssistant
    val trVecs = (io.readLines(trainFile) map { l => fe.extractFeatures(l) } toVector)
    val tstVecs = (io.readLines(testFile) map { l => fe.extractFeatures(l) } toVector)
    new LocalModelEvaluator(trVecs, tstVecs)
  }
}

object LocalModelSelector extends org.mitre.mandolin.config.LogInit {
  
  def main(args: Array[String]): Unit = {
    val appSettings = new GLPModelSettings(args) with ModelSelectionSettings   
    val builder = new MandolinModelSpaceBuilder(appSettings.modelSpace)    
    val selector = new LocalModelSelector(builder, appSettings)
    selector.search()
  }
}
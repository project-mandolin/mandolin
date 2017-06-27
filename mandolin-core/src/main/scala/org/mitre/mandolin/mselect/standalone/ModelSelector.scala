package org.mitre.mandolin.mselect.standalone

import org.mitre.mandolin.mlp._
import org.mitre.mandolin.mselect._
import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.util.LocalIOAssistant


class ModelSelector(val msb: MandolinModelSpaceBuilder, trainFile: String, testFile: Option[String], numWorkers: Int,
                    scoreSampleSize: Int, acqFunRelearnSize: Int, totalEvals: Int,
                    appSettings: Option[MandolinMLPSettings with ModelSelectionSettings] = None, useHyperband: Boolean = false, hyperMix: Float = 1.0f,
                    hyperMax: Int = 81)
  extends ModelSelectionDriver(trainFile, testFile, numWorkers, scoreSampleSize, acqFunRelearnSize, totalEvals, useHyperband, hyperMix, hyperMax) {

  def this(_msb: MandolinModelSpaceBuilder, appSettings: MandolinMLPSettings with ModelSelectionSettings) = {
    this(_msb, appSettings.trainFile.get, appSettings.testFile, appSettings.numWorkers,
      appSettings.scoreSampleSize, appSettings.updateFrequency, appSettings.totalEvals, Some(appSettings), appSettings.useHyperband,
      appSettings.hyperbandMixParam, appSettings.numEpochs)
  }

  val acqFun = appSettings match {
    case Some(s) => s.acquisitionFunction
    case None => new ExpectedImprovement
  }

  val (fe: FeatureExtractor[String, MMLPFactor], nnet: ANNetwork, numInputs: Int, numOutputs: Int, sparse: Boolean) = {
    val settings = appSettings.getOrElse((new MandolinMLPSettings).withSets(Seq(
      ("mandolin.trainer.train-file", trainFile),
      ("mandolin.trainer.test-file", testFile)
    )))

    val (trainer, nn) = MMLPTrainerBuilder(settings)
    val featureExtractor = trainer.getFe
    featureExtractor.getAlphabet.ensureFixed // fix the alphabet
    val numInputs = nn.inLayer.getNumberOfOutputs // these will then be gathered dynamically from the trainFile
    val numOutputs = nn.outLayer.getNumberOfOutputs // ditto
    val isSparse = nn.inLayer.ltype.designate match {
      case SparseInputLType => true
      case _ => false
    }
    (featureExtractor, nn, numInputs, numOutputs, isSparse)
  }

  val ms: ModelSpace = msb.build(numInputs, numOutputs, sparse, appSettings)

  val ev = {
    val io = new LocalIOAssistant
    val trVecs = (io.readLines(trainFile) map { l => fe.extractFeatures(l) } toVector)
    val tstVecs = testFile map { tf => (io.readLines(tf) map { l => fe.extractFeatures(l) } toVector) }
    new LocalModelEvaluator(trVecs, tstVecs)
  }
}

object ModelSelector extends org.mitre.mandolin.config.LogInit {

  def main(args: Array[String]): Unit = {
    val appSettings = new MandolinMLPSettings(args) with ModelSelectionSettings
    val builder = new MandolinModelSpaceBuilder(appSettings.modelSpace)
    val selector = new ModelSelector(builder, appSettings)
    selector.search()
  }
}
package org.mitre.mandolin.mselect

import org.apache.spark.SparkContext
import org.mitre.mandolin.util.LocalIOAssistant
import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.glp.{ GLPTrainerBuilder, GLPModelSettings, CategoricalGLPPredictor, GLPFactor, GLPWeights, ANNetwork, SparseInputLType }

class SparkModelSelectionDriver(val sc: SparkContext, val msb: MandolinModelSpaceBuilder, trainFile: String, testFile: String, 
    numWorkers: Int, workerBatchSize: Int, scoreSampleSize: Int, acqFunRelearnSize: Int, totalEvals: Int,
    appSettings: Option[GLPModelSettings with ModelSelectionSettings] = None) 
extends ModelSelectionDriver(trainFile, testFile, numWorkers, workerBatchSize, scoreSampleSize, acqFunRelearnSize, totalEvals) {
  
  def this(sc: SparkContext, _msb: MandolinModelSpaceBuilder, appSettings: GLPModelSettings with ModelSelectionSettings) = { 
    this(sc, _msb, appSettings.trainFile.get, appSettings.testFile.getOrElse(appSettings.trainFile.get), appSettings.numWorkers, appSettings.workerBatchSize, 
    appSettings.scoreSampleSize, appSettings.updateFrequency, appSettings.totalEvals, Some(appSettings))
  }
  
    
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
  override val ev = {
    val io = new LocalIOAssistant
    val trVecs = io.readLines(trainFile) map { l => fe.extractFeatures(l) }
    val tstVecs = io.readLines(testFile) map { l => fe.extractFeatures(l) }
    val trainBC = sc.broadcast(trVecs.toVector)
    val testBC = sc.broadcast(tstVecs.toVector)

    new SparkModelEvaluator(sc, trainBC, testBC)
  }
}


object SparkModelSelectionDriver {
  
  def main(args: Array[String]) : Unit = {
    val appSettings = new GLPModelSettings(args) with ModelSelectionSettings
    val sc = new SparkContext
    val trainFile = appSettings.trainFile.get
    val testFile = appSettings.testFile.getOrElse(trainFile)
    val builder = MandolinModelFactory.getModelSpaceBuilder(appSettings.modelSpace)    
    val selector = new SparkModelSelectionDriver(sc, builder, appSettings)
    selector.search()
  }
  
}
/*
object SparkModelSelectionDriver {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext
    //val io = new LocalIOAssistant
    val trainFile = args(0)
    val testFile = args(1)
    val numWorkers = args(2).toInt
    val numThreads = args(3).toInt
    val workerBatchSize = args(4).toInt
    val scoreSampleSize = if (args.length > 5) args(5).toInt else 240
    val acqFunRelearnSize = if (args.length > 6) args(6).toInt else 8
    val totalEvals = if (args.length > 7) args(7).toInt else 40

    val builder = MandolinLogisticRegressionFactory.getModelSpaceBuilder()
    builder.defineInitialLearningRates(0.01, 1.0)
    builder.defineOptimizerMethods("sgd", "adagrad")
    builder.defineTrainerThreads(numThreads)

    new SparkModelSelectionDriver(sc, builder, trainFile, testFile, numWorkers, workerBatchSize, scoreSampleSize, acqFunRelearnSize, totalEvals)
  }
}
*/
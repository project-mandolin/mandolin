package org.mitre.mandolin.mselect

import org.apache.spark.SparkContext
import org.mitre.mandolin.util.LocalIOAssistant

import org.mitre.mandolin.glp.{ GLPTrainerBuilder, GLPModelSettings, CategoricalGLPPredictor, GLPFactor, GLPWeights }

class SparkModelSelectionDriver(val sc: SparkContext, val msb: ModelSpaceBuilder, trainFile: String, testFile: String, 
    numWorkers: Int, workerBatchSize: Int, scoreSampleSize: Int, acqFunRelearnSize: Int, totalEvals: Int) 
extends ModelSelectionDriver(msb, trainFile, testFile, numWorkers, workerBatchSize, scoreSampleSize, acqFunRelearnSize, totalEvals) {
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
    val builder = GenericModelFactory.getModelSpaceBuilder(appSettings.modelSpace)    
    val selector = new SparkModelSelectionDriver(sc, builder, trainFile, testFile,
        appSettings.numWorkers, appSettings.workerBatchSize, appSettings.scoreSampleSize, appSettings.updateFrequency, appSettings.totalEvals)
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
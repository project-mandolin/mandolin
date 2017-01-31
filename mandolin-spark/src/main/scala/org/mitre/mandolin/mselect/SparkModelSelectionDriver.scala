package org.mitre.mandolin.mselect

import org.apache.spark.SparkContext
import org.mitre.mandolin.util.LocalIOAssistant


class SparkModelSelectionDriver(val sc: SparkContext, val ms: ModelSpace, trainFile: String, testFile: String, numWorkers: Int, numThreads: Int, workerBatchSize: Int, scoreSampleSize: Int, acqFunRelearnSize: Int, totalEvals: Int) extends ModelSelectionDriver(ms, trainFile, testFile, numWorkers, numThreads, workerBatchSize, scoreSampleSize, acqFunRelearnSize, totalEvals) {
  override val ev = {
    val io = new LocalIOAssistant
    val trVecs = io.readLines(trainFile) map { l => fe.extractFeatures(l)}
    val tstVecs = io.readLines(testFile) map { l => fe.extractFeatures(l)}
    val trainBC = sc.broadcast(trVecs.toVector)
    val testBC = sc.broadcast(tstVecs.toVector)
    new SparkModelEvaluator(sc, trainBC, testBC)
  }
}

object SparkModelSelectionDriver {

  def foo(): ModelSpace = {
    // set up model space
    val lrParam = new RealMetaParameter("lr", new RealSet(0.1, 0.95))
    val methodParam = new CategoricalMetaParameter("method", new CategoricalSet(Vector("adagrad", "sgd")))
    val trainerThreadsParam = new CategoricalMetaParameter("numTrainerThreads", new CategoricalSet(Vector("2")))
    new ModelSpace(Vector(lrParam), Vector(methodParam, trainerThreadsParam))
    // end model space
  }

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext
    val io = new LocalIOAssistant
    val trainFile = args(0)
    val testFile = args(1)
    val numWorkers = args(2).toInt
    val numThreads = args(3)
    val workerBatchSize = args(4).toInt
    val scoreSampleSize = if (args.length > 5) args(5).toInt else 240
    val acqFunRelearnSize = if (args.length > 6) args(6).toInt else 8
    val totalEvals = if (args.length > 7) args(7).toInt else 40
    new SparkModelSelectionDriver(sc, foo, trainFile, testFile, numWorkers, numThreads.toInt, workerBatchSize, scoreSampleSize, acqFunRelearnSize, totalEvals)
  }
}

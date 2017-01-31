package org.mitre.mandolin.mselect

import org.mitre.mandolin.mselect.WorkPullingPattern.RegisterWorker
import org.mitre.mandolin.util.LocalIOAssistant
import akka.actor.{PoisonPill, ActorSystem, Props}
import scala.concurrent.{ExecutionContext }
import java.util.concurrent.Executors
import org.mitre.mandolin.glp.{ GLPTrainerBuilder, GLPModelSettings, CategoricalGLPPredictor, GLPFactor, GLPWeights }
import org.mitre.mandolin.predict.local.NonExtractingEvalDecoder
import org.mitre.mandolin.predict.DiscreteConfusion

class LocalModelSelector(val ms: ModelSpace, trainFile: String, testFile: String, numWorkers: Int, workerBatchSize: Int, scoreSampleSize: Int, acqFunRelearnSize: Int, totalEvals: Int) extends ModelSelectionDriver(ms, trainFile, testFile, numWorkers, workerBatchSize, scoreSampleSize, acqFunRelearnSize, totalEvals) {
  override val ev = {
    val io = new LocalIOAssistant
    val trVecs = (io.readLines(trainFile) map { l => fe.extractFeatures(l) } toVector)
    val tstVecs = (io.readLines(testFile) map { l => fe.extractFeatures(l) } toVector)
    new LocalModelEvaluator(trVecs, tstVecs)
  }
}

object LocalModelSelector {
  
  def main(args: Array[String]): Unit = {
    val trainFile = args(0)
    val testFile = args(1)
    val numWorkers = args(2).toInt
    val numThreads = args(3)
    val workerBatchSize = args(4).toInt
    val scoreSampleSize = if (args.length > 5) args(5).toInt else 240
    val acqFunRelearnSize = if (args.length > 6) args(6).toInt else 8
    val totalEvals = if (args.length > 7) args(7).toInt else 40

    // set up model space
    val lrParam = new RealMetaParameter("lr", new RealSet(0.1, 0.95))
    val methodParam = new CategoricalMetaParameter("method", new CategoricalSet(Vector("adagrad", "sgd")))
    val trainerThreadsParam = new CategoricalMetaParameter("numTrainerThreads", new CategoricalSet(Vector(numThreads)))
    //val modelSpace = new ModelSpace(Vector(lrParam), Vector(methodParam, trainerThreadsParam), nn)
    // end model space
  }
}
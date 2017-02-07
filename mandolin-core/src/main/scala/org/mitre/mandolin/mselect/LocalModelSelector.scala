package org.mitre.mandolin.mselect

import org.mitre.mandolin.mselect.WorkPullingPattern.RegisterWorker
import org.mitre.mandolin.util.LocalIOAssistant
import akka.actor.{PoisonPill, ActorSystem, Props}
import scala.concurrent.{ExecutionContext }
import java.util.concurrent.Executors
import org.mitre.mandolin.glp.{ GLPTrainerBuilder, GLPModelSettings, CategoricalGLPPredictor, GLPFactor, GLPWeights }
import org.mitre.mandolin.predict.local.NonExtractingEvalDecoder
import org.mitre.mandolin.predict.DiscreteConfusion

class LocalModelSelector(val msb: ModelSpaceBuilder, trainFile: String, testFile: String, numWorkers: Int, workerBatchSize: Int, scoreSampleSize: Int, acqFunRelearnSize: Int, totalEvals: Int) 
extends ModelSelectionDriver(msb, trainFile, testFile, numWorkers, workerBatchSize, scoreSampleSize, acqFunRelearnSize, totalEvals) {
  override val ev = {
    val io = new LocalIOAssistant
    val trVecs = (io.readLines(trainFile) map { l => fe.extractFeatures(l) } toVector)
    val tstVecs = (io.readLines(testFile) map { l => fe.extractFeatures(l) } toVector)
    new LocalModelEvaluator(trVecs, tstVecs)
  }
}

object LocalModelSelector {
  
  def main(args: Array[String]): Unit = {
    val appSettings = new GLPModelSettings(args) with ModelSelectionSettings    
    val trainFile = appSettings.trainFile.get
    val testFile = appSettings.testFile.getOrElse(trainFile)
    val builder = GenericModelFactory.getModelSpaceBuilder(appSettings.modelSpace)    
    val selector = new LocalModelSelector(builder, trainFile, testFile, 
        appSettings.numWorkers, appSettings.workerBatchSize, appSettings.scoreSampleSize, appSettings.updateFrequency, appSettings.totalEvals)
    selector.search()
  }
}
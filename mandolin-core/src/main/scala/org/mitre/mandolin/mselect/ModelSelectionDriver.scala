package org.mitre.mandolin.mselect

import java.util.concurrent.Executors

import akka.actor.{Props, ActorSystem}
import org.mitre.mandolin.glp.{GLPTrainerBuilder, GLPModelSettings, ANNetwork, GLPFactor}
import org.mitre.mandolin.mselect.WorkPullingPattern.RegisterWorker
import org.mitre.mandolin.transform.FeatureExtractor

import scala.concurrent.ExecutionContext

/**
  * Created by jkraunelis on 1/30/17.
  */
abstract class ModelSelectionDriver(msb: ModelSpaceBuilder, trainFile: String, testFile: String, numWorkers: Int, workerBatchSize: Int, scoreSampleSize: Int, acqFunRelearnSize: Int, totalEvals: Int,
    appSettings: Option[GLPModelSettings with ModelSelectionSettings] = None) {
  val (fe: FeatureExtractor[String, GLPFactor], nnet: ANNetwork, numInputs: Int, numOutputs: Int) = {
    val settings = appSettings.getOrElse((new GLPModelSettings).withSets(Seq(
      ("mandolin.trainer.train-file", trainFile),
      ("mandolin.trainer.test-file", testFile)
    )))

    val (trainer, nn) = GLPTrainerBuilder(settings)
    val featureExtractor = trainer.getFe
    featureExtractor.getAlphabet.ensureFixed // fix the alphabet
    val numInputs = nn.inLayer.getNumberOfOutputs // these will then be gathered dynamically from the trainFile
    val numOutputs = nn.outLayer.getNumberOfOutputs // ditto
    (featureExtractor, nn, numInputs, numOutputs)
  }

  // now build the model space after we know the number of inputs and outptus from reading in training set
  val ms: ModelSpace = msb.build(numInputs, numOutputs)
  val ev: ModelEvaluator   

  def search( ) : Unit = {

    implicit val ec = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numWorkers))
    val system = ActorSystem("ModelSelectionActorSystem")
    val master = system.actorOf(Props(new ModelConfigEvaluator[ModelConfig]), name = "master")
    val acqFun = new BayesianNNAcquisitionFunction(ms)
    val scorerActor = system.actorOf(Props(new ModelScorer(ms, acqFun, master, scoreSampleSize, acqFunRelearnSize, totalEvals)), name = "scorer")
    val workers = 1 to numWorkers map (i => system.actorOf(Props(new ModelConfigEvalWorker(master, scorerActor, ev, workerBatchSize)), name = "worker" + i))
    Thread.sleep(2000)
    workers.foreach(worker => master ! RegisterWorker(worker))
  }
}

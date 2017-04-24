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
abstract class ModelSelectionDriver(trainFile: String, testFile: String, numWorkers: Int,
                                    scoreSampleSize: Int, acqFunRelearnSize: Int, 
                                    totalEvals: Int, hyperBand: Boolean = false, hyperMix: Float = 1.0f,
                                    hyperMax: Int = 81) {

  val ms: ModelSpace
  val acqFun : AcquisitionFunction
  val ev: ModelEvaluator

  def search(): Unit = {
    implicit val ec = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numWorkers))
    val system = ActorSystem("ModelSelectionActorSystem")
    val master = system.actorOf(Props(new ModelConfigEvaluator[ModelConfig]), name = "master")    
    val scoringFun = 
      if (acqFun.isInstanceOf[RandomAcquisition]) new RandomScoringFunction else 
      new BayesianNNScoringFunction(ms, acqFun, numWorkers)
    val scorer = 
      if (hyperBand) Props(new HyperbandModelScorer(ms, scoringFun, master, scoreSampleSize, acqFunRelearnSize, totalEvals, numWorkers, hyperMax, hyperMix))
      else Props(new ModelScorer(ms, scoringFun, master, scoreSampleSize, acqFunRelearnSize, totalEvals, numWorkers))
    val scorerActor = system.actorOf(scorer, name = "scorer")
    val workers = 1 to numWorkers map (i => 
      system.actorOf(Props(new ModelConfigEvalWorker(master, scorerActor, ev)), name = "worker" + i))
    Thread.sleep(2000) // TODO this is a workaround for slow joining workers - revisit
    workers.foreach(worker => master ! RegisterWorker(worker))
  }
}

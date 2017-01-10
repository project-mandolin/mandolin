package org.mitre.mandolin.mselect

import akka.actor.{ActorRef, Actor}
import org.slf4j.LoggerFactory

/**
  * Encapsulates functionality to apply Bayesian regression model to "score"
  * model configurations - where a score is the estimated performance/accuracy/error
  * of the model on a given dataset.
  */
class ModelScorer(modelConfigSpace: ModelSpace, acqFn: AcquisitionFunction, evalMaster: ActorRef, sampleSize: Int, acqFnThreshold: Int) extends Actor {
  
  //def this(mcs: ModelSpace, af: AcquisitionFunction) = this(mcs, af, 10, 10)
    
  import WorkPullingPattern._

  val log = LoggerFactory.getLogger(getClass)
  var evalResults = new collection.mutable.ArrayBuffer[ModelEvalResult]
  var receivedSinceLastScore = 0

  override def preStart() = {
    val scored = getScoredConfigs(sampleSize) map ( _._2 )
    val epic = new Epic[ModelConfig] {
      override val iterator = scored.toIterator
    }
    evalMaster ! epic
  }

  // should receive messages sent from ModelConfigEvaluator
  def receive = {
    case ModelEvalResult(ms, res) =>
      log.info("Received score " + res + " from model " + ms)
      evalResults append ModelEvalResult(ms, res)
      receivedSinceLastScore += 1
      if (receivedSinceLastScore > acqFnThreshold) {
        log.info("Training acquisition function")
        receivedSinceLastScore = 0
        acqFn.train(evalResults)
        log.info("Finished training acquisition function")
        
        val scored = getScoredConfigs(sampleSize) map {_._2}
        val epic = new Epic[ModelConfig] {override val iterator = scored.toIterator}
        evalMaster ! epic
      }

    /*
      case ProvideWork => // means that model evaluation is ready to evaluation models
      log.info("Received ProvideWork")
      val scored = getScoredConfigs(sampleSize) map {_._2}
      val epic = new Epic[ModelConfig] {override val iterator = scored.toIterator}
      sender ! epic
      */
  }

  def getScoredConfigs(size: Int) = {
    val unscoredConfigs = for (i <- 1 to size) yield modelConfigSpace.drawRandom
    (unscoredConfigs map { s => (acqFn.score(s), s) }).toVector.sortWith((a, b) => a._1 > b._1)
  }
}
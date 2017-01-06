package org.mitre.mandolin.mselect

import akka.actor.Actor
import org.mitre.mandolin.glp.GLPModelSpec
import org.slf4j.LoggerFactory

/**
 * Encapsulates functionality to apply Bayesian regression model to "score"
 * model configurations - where a score is the estimated performance/accuracy/error
 * of the model on a given dataset.
 */
class ModelScorer(modelConfigSpace: ModelSpace, acqFn: AcquisitionFunction, sampleSize: Int, acqFnThreshold: Int) extends Actor {
    
  import WorkPullingPattern._

  val log = LoggerFactory.getLogger(getClass)
  var evalResults = new collection.mutable.ArrayBuffer[ModelEvalResult]
  var receivedSinceLastScore = 0

  // should receive messages sent from ModelConfigEvaluator
  def receive = {
    case ModelEvalResult(ms, res) => 
      log.info("Received score " + res + " from model " + ms)
      evalResults append ModelEvalResult(ms,res)
      receivedSinceLastScore += 1
      
      if (receivedSinceLastScore > acqFnThreshold) {
        log.info("Training acquisition function")
        receivedSinceLastScore = 0
        acqFn.train(evalResults)
        log.info("Finished training acquisition function")
        
        val scored = getScoredConfigs(sampleSize) map {_._2}
        val epic = new Epic[ModelConfig] {override val iterator = scored.toIterator}
        sender ! epic
      }

    case ProvideWork => // means that model evaluation is ready to evaluation models
      log.info("Received ProvideWork")
      val scored = getScoredConfigs(sampleSize) map {_._2}
      val epic = new Epic[ModelConfig] {override val iterator = scored.toIterator}
      sender ! epic
  }
  
  def getScoredConfigs(size: Int) = {
    val unscoredConfigs = for (i <- 1 to size) yield modelConfigSpace.drawRandom
    (unscoredConfigs map {s => (acqFn.score(s),s)}).toVector.sortWith((a,b) => a._1 > b._1)
  }
      
}
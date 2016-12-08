package org.mitre.mandolin.mselect

import akka.actor.Actor
import org.mitre.mandolin.glp.GLPModelSpec

/**
 * Encapsulates functionality to apply Bayesian regression model to "score"
 * model configurations - where a score is the estimated performance/accuracy/error
 * of the model on a given dataset.
 */
class ModelScorer(modelConfigSpace: ModelSpace, acqFn: AcquisitionFunction) extends Actor {
    
  import WorkPullingPattern._
  
  val frequency = 10
  var evalResults = new collection.mutable.ArrayBuffer[ModelEvalResult]
  var receivedSinceLastScore = 0
  var currentSampleSize = 5

  // should receive messages sent from ModelConfigEvaluator
  def receive = {
    case ModelEvalResult(ms, res) => 
      println("Received score " + res + " from model " + ms)
      evalResults append ModelEvalResult(ms,res)
      receivedSinceLastScore += 1
      /*
      if (receivedSinceLastScore > 10) {
        receivedSinceLastScore = 0
        val mspec = buildNewScoringModel()
        currentModel = Some(mspec)
        applyModel(mspec)
      }
      * 
      */
    case ProvideWork => // means that model evaluation is ready to evaluation models
      println("ModelScorer**** ==> Received ProvideWork")
      val scored = getScoredConfigs(currentSampleSize) map {_._2}
      println("Scored vector length = " + scored.length)
      val epic = new Epic[ModelConfig] {override val iterator = scored.toIterator}
      sender ! epic
  }
  
  def getScoredConfigs(size: Int) = {
    val unscoredConfigs = for (i <- 1 to size) yield modelConfigSpace.drawRandom
    (unscoredConfigs map {s => (acqFn.score(s),s)}).toVector.sortWith((a,b) => a._1 > b._1)
  }
      
}
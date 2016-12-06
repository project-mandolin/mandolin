package org.mitre.mandolin.mselect

import akka.actor.Actor
import org.mitre.mandolin.glp.GLPModelSpec

/**
 * Encapsulates functionality to apply Bayesian regression model to "score"
 * model configurations - where a score is the estimated performance/accuracy/error
 * of the model on a given dataset.
 */
class ModelScorer extends Actor {
    
  import WorkPullingPattern._
  
  val frequency = 10
  var evalResults = new collection.mutable.ArrayBuffer[ModelEvalResult]
  var receivedSinceLastScore = 0
  var currentModel : Option[GLPModelSpec] = None 

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
  }
    
  def applyModel(mspec: GLPModelSpec) = {
    // get candidate model configurations
    val candidateConfigs : List[String] = Nil
    
  }
  
  def buildNewScoringModel() : GLPModelSpec = {
    // take evalResults and train a GLP (should be very quick, but must consider case where this is time consuming and spawn thread)
    throw new RuntimeException("Not implemented yet")
  }
  
}
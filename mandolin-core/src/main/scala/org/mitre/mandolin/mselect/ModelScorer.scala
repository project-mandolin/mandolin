package org.mitre.mandolin.mselect

import akka.actor.{ActorRef, Actor}
import org.slf4j.LoggerFactory
import org.mitre.mandolin.util.Alphabet

case class ScoredModelConfig(sc: Double, mc: ModelConfig)


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
  val startTime = System.currentTimeMillis()

  override def preStart() = {
    val scored = getScoredConfigs(sampleSize) map ( _._2 )
    val epic = new Epic[ModelConfig] {
      override val iterator = scored.toIterator
    }
    evalMaster ! epic
  }

  // should receive messages sent from ModelConfigEvaluator
  def receive = {
    case ModelEvalResult(r)  =>

      evalResults append ModelEvalResult(r)
      receivedSinceLastScore += 1
      if (receivedSinceLastScore == sampleSize) {
        val hours = System.currentTimeMillis() - startTime /1000 /60 /60
        log.info(s"Total time for $sampleSize configs was $hours hours")
        System.exit(0)
      }
      if (receivedSinceLastScore > acqFnThreshold) {
        log.info("Training acquisition function")
        receivedSinceLastScore = 0
        acqFn.train(evalResults)
        log.info("Finished training acquisition function")        
        val scored = getScoredConfigs(sampleSize) map {_._2}
        val epic = new Epic[ModelConfig] {override val iterator = scored.toIterator}
        evalMaster ! epic
      }
  }
  
  

  def getScoredConfigs(size: Int) = {
    val unscoredConfigs = for (i <- 1 to size) yield modelConfigSpace.drawRandom
    (unscoredConfigs map { s => (acqFn.score(s), s) }).toVector.sortWith((a, b) => a._1 > b._1)
  }
}
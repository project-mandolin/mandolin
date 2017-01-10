package org.mitre.mandolin.mselect

import akka.actor.{ActorRef, Actor}
import org.mitre.mandolin.glp.GLPModelSpec
import org.slf4j.LoggerFactory

/**
  * Encapsulates functionality to apply Bayesian regression model to "score"
  * model configurations - where a score is the estimated performance/accuracy/error
  * of the model on a given dataset.
  */
class ModelScorer(modelConfigSpace: ModelSpace, acqFn: AcquisitionFunction, evalMaster: ActorRef) extends Actor {

  import WorkPullingPattern._

  val log = LoggerFactory.getLogger(getClass)
  val frequency = 10
  var evalResults = new collection.mutable.ArrayBuffer[ModelEvalResult]
  var receivedSinceLastScore = 0
  var currentSampleSize = 10000
  var sendMoreThreshold = 10001

  override def preStart() = {
    val scored = getScoredConfigs(currentSampleSize) map ( _._2 )
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

      if (receivedSinceLastScore > 40) {
        log.info("Received " + sendMoreThreshold + " results, sending more configs")
        receivedSinceLastScore = 0
        val scored = getScoredConfigs(currentSampleSize) map ( _._2 )
        val epic = new Epic[ModelConfig] {
          override val iterator = scored.toIterator
        }
        evalMaster ! epic
        //val mspec = buildNewScoringModel()
        //currentModel = Some(mspec)
        //applyModel(mspec)
      }

    /*case ProvideWork(numConfigs) => // means that model evaluation is ready to evaluation models
        log.info("Received ProvideWork(" + numConfigs + ")")
        val scored = getScoredConfigs(currentSampleSize) map {
          _._2
        }
        val epic = new Epic[ModelConfig] {
          override val iterator = scored.toIterator
        }
        sender ! epic
    }*/
  }

  def getScoredConfigs(size: Int) = {
    val unscoredConfigs = for (i <- 1 to size) yield modelConfigSpace.drawRandom
    (unscoredConfigs map { s => (acqFn.score(s), s) }).toVector.sortWith((a, b) => a._1 > b._1)
  }
}
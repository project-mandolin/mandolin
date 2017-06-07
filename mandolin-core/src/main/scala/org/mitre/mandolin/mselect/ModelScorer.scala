package org.mitre.mandolin.mselect

import java.io.{File, PrintWriter}
import java.text.SimpleDateFormat
import java.util.Calendar

import akka.actor.{ActorRef, Actor}
import org.slf4j.LoggerFactory
import org.mitre.mandolin.util.Alphabet

case class ScoredModelConfig(sc: Double, mc: ModelConfig, src: Int = 0)


/**
  * Encapsulates functionality to apply Bayesian regression model to "score"
  * model configurations - where a score is the estimated performance/accuracy/error
  * of the model on a given dataset.
  */
class ModelScorer(modelConfigSpace: ModelSpace, acqFn: ScoringFunction, evalMaster: ActorRef,
                  sampleSize: Int, acqFnThreshold: Int, totalEvals: Int, concurrentEvals: Int) extends Actor {

  import WorkPullingPattern._

  val log = LoggerFactory.getLogger(getClass)
  val now = Calendar.getInstance.getTime
  val outWriter = new PrintWriter(new File("mselect-" + new SimpleDateFormat("yyyyMMdd-HHmmss").format(now) + ".csv"))
  var evalResults = new collection.mutable.ArrayBuffer[ScoredModelConfig]
  // keep track of models currently being evaluated
  var currentlyEvaluating : Set[ModelConfig] = Set()
  var receivedSinceLastScore = 0
  val startTime = System.currentTimeMillis()
  protected val startWallTime = System.nanoTime

  def totalReceived = evalResults.length

  override def preStart() = {
    // send initial "random batch" of configs to evaluate
    val scored = getScoredConfigs(sampleSize) map (_._2)
    val epic = new Epic[ModelConfig] {
      override val iterator = scored.toIterator
    }
    evalMaster ! epic
    log.info("SCORER: Finished pre-start")
  }

  // should receive messages sent from ModelConfigEvaluator
  def receive = {
    case ModelEvalResult(r) =>
      currentlyEvaluating -= r.mc   // remove this from set of currently evaluating model configs
      evalResults += r
      receivedSinceLastScore += 1
      log.info("accuracy:" + r.sc + " " + r.mc + "\n")
      outWriter.print("accuracy:" + r.sc + " cumulativeTime:" + ((System.nanoTime() - startWallTime) / 1E9) + " " + r.mc + "\n")
      outWriter.flush()
      if (totalReceived >= totalEvals) {
        outWriter.close()
        val hours = (System.currentTimeMillis() - startTime) / 1000.0 / 60 / 60
        log.info(s"Total time for $totalEvals model evaluations was $hours hours")
        System.exit(0)
      }
      if (receivedSinceLastScore >= acqFnThreshold) {
        log.info("Training acquisition function")
        receivedSinceLastScore = 0
        acqFn.train(evalResults)  
        log.info("Finished training acquisition function")
        val scored = getScoredConfigs(sampleSize)
        log.info("Building new batch to evaluate based on scores [top 10]: ")
        scored.take(10) foreach { case (v, c) => log.info("score: " + v) }
        val configs = scored map {
          _._2
        }
        val epic = new Epic[ModelConfig] {
          override val iterator = configs.toIterator
        }
        evalMaster ! epic
      }
    case CurrentlyEvaluating(c) => currentlyEvaluating += c
    case Hello => log.info("SCORER: Received Hello from " + sender.toString())
  }

  def getScoredConfigs(size: Int) = {
    val unscoredConfigs = for (i <- 1 to size) yield modelConfigSpace.drawRandom
    if (concurrentEvals > 1) {
      // actually do a full concurrent scoring of the number of concurrent evaluations + number of evals needed to rebuild the acquisition
      // function
      val numToScoreConcurrent = concurrentEvals + acqFnThreshold * 2
      acqFn.scoreConcurrent(unscoredConfigs.toVector, numToScoreConcurrent)
    } else {      
      (unscoredConfigs map { s => (acqFn.score(s), s) }).toVector.sortWith((a, b) => a._1 > b._1)
    }
  }
}
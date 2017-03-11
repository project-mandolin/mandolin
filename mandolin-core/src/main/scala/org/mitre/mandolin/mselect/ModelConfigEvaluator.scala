package org.mitre.mandolin.mselect

import java.util.concurrent.Executors

import akka.actor.Actor
import akka.actor.ActorRef
import org.slf4j.LoggerFactory
import scala.concurrent.{ExecutionContext, ExecutionContextExecutor, Future}

object WorkPullingPattern {

  sealed trait Message

  trait Epic[T] extends Iterable[T]

  //used by master to create work (in a streaming way)
  case class ProvideWork(batchSize: Int) extends Message

  case object CurrentlyBusy extends Message

  case object WorkAvailable extends Message

  case class RegisterWorker(worker: ActorRef) extends Message

  case class Terminated(worker: ActorRef) extends Message

  case class Work[T](work: T) extends Message

  case class Update(acquisitionFunction: AcquisitionFunction) extends Message
  
  case object Hello extends Message

  // config should end up being a model config/specification
  case class ModelEvalResult(configResults: Seq[ScoredModelConfig]) extends Message

}

/**
  * This will take configurations from the ModelConfigQueue
  * It will act as master with workers carrying out the full evaluation
  */
class ModelConfigEvaluator[T]() extends Actor {

  import WorkPullingPattern._

  val log = LoggerFactory.getLogger(getClass)
  val workers = collection.mutable.Set.empty[ActorRef]
  var currentEpic: Option[Epic[T]] = None

  def receive = {
    case epic: Epic[T] =>
      if (workers.isEmpty) {
        log.error("Got work but there are no workers registered")
        //System.exit(0)
      }
      log.info("Got new epic from ModelScorer")
      currentEpic = Some(epic)
      log.info("Telling workers there is work available")
      workers foreach {
        _ ! WorkAvailable
      }

    case RegisterWorker(worker) =>
      log.info(s"worker $worker registered")
      context.watch(worker)
      workers += worker
      if (currentEpic.isDefined) worker ! WorkAvailable

    case Terminated(worker) =>
      log.info(s"worker $worker died - taking off the set of workers")
      workers.remove(worker)

    case ProvideWork(batchSize) => currentEpic match {
      case None =>
        log.info("workers asked for work but we've no more work to do")
      //scorer ! ProvideWork(numConfigs) // request work from the scorer
      case Some(epic) ⇒
        log.info(s"Received ProvideWork($batchSize), checking epic")
        val iter = epic.iterator
        val batch = (1 to batchSize).map { i =>
          if (currentEpic.isDefined && iter.hasNext) {
            Some(iter.next())
          } else {
            log.info(s"done with current epic $epic")
            currentEpic = None
            None
          }
        }.filter(_.isDefined).map(_.get)
        log.info(s"Sending batch of size ${batch.length} to worker $sender")
        if (batch.length > 0) sender ! Work(batch)

    }

    case x => log.info("Received unrecognized message " + x.toString)
  }

}

/**
  * This worker actor will actually take work from the master in the form of models to evaluate
  */
class ModelConfigEvalWorker(val master: ActorRef, val modelScorer: ActorRef, modelEvaluator: ModelEvaluator, batchSize: Int) extends Actor {

  var busy : Boolean = false

  import WorkPullingPattern._

  //import scala.concurrent.ExecutionContext.Implicits.global
  
  implicit val ec = new ExecutionContext {
    val threadPool = Executors.newFixedThreadPool(128);

    def execute(runnable: Runnable) {
        threadPool.submit(runnable)
    }

    def reportFailure(t: Throwable) {}
  }
  
  val log = LoggerFactory.getLogger(getClass)

  def receive = {
    case WorkAvailable => {
      log.info(s"Worker $this received work available, asking master to provide work")
      if (!busy) master ! ProvideWork(batchSize)
    }
    case Work(w: Seq[ModelConfig]) =>
      log.info(s"Worker $this about to do work...")
      doWork(w) onComplete { case r =>
        log.info(s"Worker $this finished configuration; sending result of " + r.get.configResults.seq(0).sc + " to " + modelScorer)
        modelScorer ! r.get // send result to modelScorer
        busy = false
        master ! ProvideWork(batchSize)
      }
    case x => log.error("Received unrecognized message " + x)
  }

  def doWork(w: Seq[ModelConfig]): Future[ModelEvalResult] = {
    busy = true
    Future({
      val score = modelEvaluator.evaluate(w)
      // actually get the model configs evaluation result
      // send to modelScorer
      log.info("Scores: " + score.mkString(" "))
      ModelEvalResult(score zip w map {case (s,c) => ScoredModelConfig(s,c)})
    })
  }
}


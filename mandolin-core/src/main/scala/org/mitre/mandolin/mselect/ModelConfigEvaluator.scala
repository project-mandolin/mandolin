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
  case object ProvideWork extends Message

  case object CurrentlyBusy extends Message

  case object WorkAvailable extends Message

  case class RegisterWorker(worker: ActorRef) extends Message

  case class Terminated(worker: ActorRef) extends Message

  case class Work[T](work: T, generation: Int) extends Message

  case class CancelTraining(generation: Int) extends Message

  case class Update(acquisitionFunction: AcquisitionFunction) extends Message
  
  case object Hello extends Message

  case class CurrentlyEvaluating(c: ModelConfig) extends Message
  
  // config should end up being a model config/specification
  case class ModelEvalResult(configResults: ScoredModelConfig) extends Message

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
  var generation : Int = 0

  def receive = {
    case epic: Epic[T] =>
      if (workers.isEmpty) {
        log.error("Got work but there are no workers registered")
        //System.exit(0)
      }
      log.info("Got new epic from ModelScorer")
      //workers foreach {
      //  _ ! CancelTraining(generation)
      //}
      currentEpic = Some(epic)
      generation += 1
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

    case ProvideWork => currentEpic match {
      case None =>
        log.info("workers asked for work but we've no more work to do")
      //scorer ! ProvideWork(numConfigs) // request work from the scorer
      case Some(epic) ⇒
        log.info(s"Received ProvideWork, checking epic")
        val iter = epic.iterator
        val batch = 
          if (currentEpic.isDefined && iter.hasNext) {
            Some(iter.next())
          } else {
            log.info(s"done with current epic $epic")
            currentEpic = None
            None
          }
        log.info(s"Sending work to worker $sender")
        batch foreach {b => sender ! Work(b, generation) }
    }

    case x => log.info("Received unrecognized message " + x.toString)
  }

}

/**
  * This worker actor will actually take work from the master in the form of models to evaluate
  */
class ModelConfigEvalWorker(val master: ActorRef, val modelScorer: ActorRef, modelEvaluator: ModelEvaluator) extends Actor {

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
      if (!busy) master ! ProvideWork
    }
    case Work(w: ModelConfig, gen: Int) =>
      log.info(s"Worker $this about to do work...")
      modelScorer ! CurrentlyEvaluating(w)
      doWork(w, gen) onComplete { case r =>
        log.info(s"Worker $this finished configuration; sending result of " + r.get.configResults.sc + " to " + modelScorer)
        modelScorer ! r.get // send result to modelScorer
        busy = false
        master ! ProvideWork
      }
    case CancelTraining(gen: Int) =>
      modelEvaluator.cancel(gen)
      busy = false
    case x => log.error("Received unrecognized message " + x)
  }

  def doWork(w: ModelConfig, gen: Int): Future[ModelEvalResult] = {
    busy = true
    Future({
      log.info("About to evaluate model ...")
      val (acc, time): (Double, Long) = modelEvaluator.evaluate(w)
      // actually get the model configs evaluation result
      // send to modelScorer
      log.info("Acc: " + acc + " Time: " + time)
      ModelEvalResult(ScoredModelConfig(acc, time, w))
    })
  }
}


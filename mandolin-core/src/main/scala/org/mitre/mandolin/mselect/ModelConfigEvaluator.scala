package org.mitre.mandolin.mselect

import akka.actor.Actor
import akka.actor.ActorRef
import scala.reflect.ClassTag
import org.slf4j.LoggerFactory
import scala.concurrent.Future

object WorkPullingPattern {
  sealed trait Message
  trait Epic[T] extends Iterable[T] //used by master to create work (in a streaming way)
  case class ProvideWork(numConfigs: Int) extends Message
  case object CurrentlyBusy extends Message
  case object WorkAvailable extends Message
  case class RegisterWorker(worker: ActorRef) extends Message
  case class Terminated(worker: ActorRef) extends Message
  case class Work[T](work: T) extends Message
  // config should end up being a model config/specification
  case class ModelEvalResult(config: ModelConfig, result: Double) extends Message
}

/**
 * THis will take configurations from the ModelConfigQueue
 * It will act as master with workers carrying out the full evaluation
 */
class ModelConfigEvaluator[T](scorer: ActorRef) extends Actor {
  import WorkPullingPattern._
  val log = LoggerFactory.getLogger(getClass)
  val workers = collection.mutable.Set.empty[ActorRef]
  var currentEpic: Option[Epic[T]] = None

  def receive = {
     case epic: Epic[T] ⇒
      if (currentEpic.isDefined)
        sender ! CurrentlyBusy
      else if (workers.isEmpty)
        log.error("Got work but there are no workers registered.")
      else {
        currentEpic = Some(epic)
        workers foreach { _ ! WorkAvailable }
      }

    case RegisterWorker(worker) ⇒
      log.info(s"worker $worker registered")
      context.watch(worker)
      workers += worker

    case Terminated(worker) ⇒
      log.info(s"worker $worker died - taking off the set of workers")
      workers.remove(worker)

    case ProvideWork(numConfigs) ⇒ currentEpic match {
      case None ⇒
        log.info("workers asked for work but we've no more work to do")
        scorer ! ProvideWork(numConfigs) // request work from the scorer
      case Some(epic) ⇒
        val iter = epic.iterator
        if (iter.hasNext)
          sender ! Work(iter.next)
        }
        else {
          log.info(s"done with current epic $epic")
          currentEpic = None
        }
    }

    case _ =>
  }
}

/**
 * This worker actor will actually take work from the master in the form of models to evaluate
 */
class ModelConfigEvalWorker(val master: ActorRef, modelScorer: ActorRef, modelEvaluator: ModelEvaluator) extends Actor {
  import WorkPullingPattern._
  implicit val ec = context.dispatcher

  def receive = {
    case WorkAvailable => master ! ProvideWork
    case Work(w: ModelConfig) => doWork(w) onComplete { case r =>
      println("Worker " + this + " processing configuration ....")
      modelScorer ! r.get // send result to modelScorer
      master ! ProvideWork }
  }

  def doWork(w: ModelConfig) : Future[ModelEvalResult] = {
    Future({
      val score = modelEvaluator.evaluate(w)
      // actually get the model configs evaluation result
      // send to modelScorer
      ModelEvalResult(w, score)
    })
  }
}


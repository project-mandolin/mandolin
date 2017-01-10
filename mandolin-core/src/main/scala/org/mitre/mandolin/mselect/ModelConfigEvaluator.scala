package org.mitre.mandolin.mselect

import akka.actor.Actor
import akka.actor.ActorRef
import scala.collection.immutable.IndexedSeq
import scala.reflect.ClassTag
import org.slf4j.LoggerFactory
import scala.concurrent.Future

object WorkPullingPattern {

  sealed trait Message

  trait Epic[T] extends Iterable[T]

  //used by master to create work (in a streaming way)
  case class ProvideWork(numConfigs: Int) extends Message

  case object CurrentlyBusy extends Message

  case object WorkAvailable extends Message

  case class RegisterWorker(worker: ActorRef) extends Message

  case class Terminated(worker: ActorRef) extends Message

  case class Work[T](work: T) extends Message

  // config should end up being a model config/specification
  case class ModelEvalResult(config: Seq[ModelConfig], result: Seq[Double]) extends Message

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
    case epic: Epic[T] ⇒
      if (workers.isEmpty) {
        log.error("Got work but there are no workers registered")
        //System.exit(0)
      }
      log.info("Got new epic from ModelScorer")
      currentEpic = Some(epic)
    //log.info("Telling workers there is work available")
    //workers foreach {
    //  _ ! WorkAvailable
    //}

    case RegisterWorker(worker) ⇒
      log.info(s"worker $worker registered")
      context.watch(worker)
      workers += worker
      worker ! WorkAvailable

    case Terminated(worker) ⇒
      log.info(s"worker $worker died - taking off the set of workers")
      workers.remove(worker)

    case ProvideWork(numConfigs) ⇒ currentEpic match {
      case None ⇒
        log.info("workers asked for work but we've no more work to do")
      //scorer ! ProvideWork(numConfigs) // request work from the scorer
      case Some(epic) ⇒
        log.info("Received ProvideWork(" + numConfigs + "), checking epic")
        val iter = epic.iterator
        val batch = (1 to numConfigs).map { i =>
          if (currentEpic.isDefined && iter.hasNext) {
            Some(iter.next())
          } else {
            log.info(s"done with current epic $epic")
            currentEpic = None
            None
          }
        }.filter(_.isDefined).map(_.get)
        log.info(s"Sending batch of size ${batch.length} to worker $sender")
        sender ! Work(batch)

    }

    case x => log.info("Received unrecognized message " + x.toString)
  }

}

/**
  * This worker actor will actually take work from the master in the form of models to evaluate
  */
class ModelConfigEvalWorker(val master: ActorRef, modelScorer: ActorRef, modelEvaluator: ModelEvaluator, batchSize: Int) extends Actor {

  import WorkPullingPattern._

  implicit val ec = context.dispatcher

  val log = LoggerFactory.getLogger(getClass)

  def receive = {
    case WorkAvailable => {
      log.info(s"Worker $this received work available, asking master to provide work")
      master ! ProvideWork(batchSize)
    }
    case Work(w: Seq[ModelConfig]) => doWork(w) onComplete { case r =>
      log.info(s"Worker $this processing configuration")
      modelScorer ! r.get // send result to modelScorer
      master ! ProvideWork(batchSize)
    }
    case x => log.error("Received unrecognized message " + x)
  }

  def doWork(w: Seq[ModelConfig]): Future[ModelEvalResult] = {
    Future({
      val score = modelEvaluator.evaluate(w)
      // actually get the model configs evaluation result
      // send to modelScorer
      ModelEvalResult(w, score)
    })
  }
}


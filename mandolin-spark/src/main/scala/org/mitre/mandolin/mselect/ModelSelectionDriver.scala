package org.mitre.mandolin.mselect

import org.apache.spark.SparkContext
import org.mitre.mandolin.mselect.WorkPullingPattern.{ProvideWork, RegisterWorker}
import org.mitre.mandolin.util.LocalIOAssistant
import akka.actor.{PoisonPill, ActorSystem, Props}

/**
  * Created by jkraunelis on 1/1/17.
  */
object ModelSelectionDriver {
  //val modelSpace = new ModelSpace(Vector(new RealMetaParameter("lr", new RealSet(0.1, 1.0))), Vector())
  val t: Vector[String] = Vector("adagrad", "sgd")
  val modelSpace = new ModelSpace(Vector(), Vector(new CategoricalMetaParameter("method", new CategoricalSet(t))))

  def getRandomConfigs = for (i <- 1 to 100) yield modelSpace.drawRandom

  def main(args: Array[String]) : Unit = {
    val system = ActorSystem("Simulator")

    val scorerActor = system.actorOf(Props(new ModelScorer(modelSpace, new RandomAcquisitionFunction, 10, 10)), name = "scorer")
    val sc = new SparkContext
    val io = new LocalIOAssistant
    val trainFile = args(0)
    val trainBC = sc.broadcast(io.readLines(trainFile).toVector)
    val ev = new SparkModelEvaluator(sc, trainBC, trainBC)
    val master = system.actorOf(Props(new ModelConfigEvaluator[ModelConfig](scorerActor)), name = "master")
    val worker1 = system.actorOf(Props(new ModelConfigEvalWorker(master, scorerActor, ev)), name = "worker1")
    //val worker2 = system.actorOf(Props(new ModelConfigEvalWorker(master, scorerActor, ev)), name = "worker2")
    //val worker3 = system.actorOf(Props(new ModelConfigEvalWorker(master, scorerActor, ev)), name = "worker3")

    master ! RegisterWorker(worker1)
    //master ! RegisterWorker(worker2)
    //master ! RegisterWorker(worker3)

    master ! ProvideWork // this starts things off
    // master should request work from the scorer if it doesn't have any

    Thread.sleep(1000*60*30)

    worker1 ! PoisonPill
    //worker2 ! PoisonPill
    //worker3 ! PoisonPill

    scorerActor ! PoisonPill
    master ! PoisonPill



  }

}

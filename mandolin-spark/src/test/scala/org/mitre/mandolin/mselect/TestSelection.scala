package org.mitre.mandolin.mselect

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.scalatest._
import akka.actor.{ActorSystem, Props}

class TestSelection extends FlatSpec with Matchers {

  import akka.actor.PoisonPill
  import WorkPullingPattern._

  val modelSpace = new ModelSpace(Vector(new RealMetaParameter("lr",new RealSet(0.1,1.0))), Vector())
  def getRandomConfigs = for (i <- 1 to 100) yield modelSpace.drawRandom

  "A random evaluator simulation" should "evaluate concurrently" in {
    val system = ActorSystem("Simulator")


    val numWorkers = 3
    val sc = new SparkContext()
    val nt : Broadcast[Vector[String]] = sc.broadcast(Vector())
    val ntst : Broadcast[Vector[String]] = sc.broadcast(Vector())
    val ev = new SparkModelEvaluator(sc, nt, ntst)
    
    val master = system.actorOf(Props(new ModelConfigEvaluator[ModelConfig]()), name = "master")
    val scorerActor = system.actorOf(Props(new ModelScorer(modelSpace, new MockAcquisitionFunction, master, 100, 2)), name = "scorer")

    
    val workers = for (i <- 1 to numWorkers) yield system.actorOf(Props(new ModelConfigEvalWorker(master, scorerActor, ev, 1)), name = "worker"+i)

    workers foreach { master ! RegisterWorker(_) }

    master ! ProvideWork // this starts things off
    // master should request work from the scorer if it doesn't have any

    Thread.sleep(30000) // wait 30 seconds


    scorerActor ! PoisonPill
    master ! PoisonPill


    /*
    val res = scorer.evalResults.toVector
    println("Got results...")
    res foreach {r => println(r.result)}
    (assert(res.length == confs.size))
    *
    */
  }
}
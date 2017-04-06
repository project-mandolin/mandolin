package org.mitre.mandolin.mselect

import scala.Vector

import org.scalatest.FlatSpec
import org.scalatest.Matchers

import akka.actor.ActorSystem
import akka.actor.Props
import akka.actor.actorRef2Scala

import org.slf4j.LoggerFactory

class TestSelection extends FlatSpec with Matchers {
  
  import WorkPullingPattern._
  import akka.actor.PoisonPill
  
  val modelSpace = new ModelSpace(Vector(new RealMetaParameter("lr",new RealSet(0.1,1.0))), Vector(), Vector())
  
  "A random evaluator simulation" should "evaluate concurrently" in {
    val log = LoggerFactory.getLogger(getClass)
    log.info("Starting simulation")
    
    val system = ActorSystem("Simulator")

    val numWorkers = 10
    val master = system.actorOf(Props(new ModelConfigEvaluator[ModelConfig]()), name = "master")
    val scoring = new BayesianNNScoringFunction(modelSpace, new RandomAcquisition)
    val scorerActor = system.actorOf(Props(new ModelScorer(modelSpace, scoring, master, 100, 2, 100)), name = "scorer")
    val ev = new MockRandomModelEvaluator
    
    val workers = for (i <- 1 to numWorkers) yield system.actorOf(Props(new ModelConfigEvalWorker(master, scorerActor, ev, 1)), name = "worker"+i)

    
    workers foreach {w => master ! RegisterWorker(w)}
    Thread.sleep(1000)
    
    master ! ProvideWork // this starts things off
    // master should request work from the scorer if it doesn't have any

    Thread.sleep(30000) // wait 30 seconds
    
    workers foreach { _ ! PoisonPill}    
    
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
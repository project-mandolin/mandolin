package org.mitre.mandolin.mselect

import org.scalatest._
import akka.actor.{ActorSystem, Props}

class TestSelection extends FlatSpec with Matchers {
  
  import akka.actor.PoisonPill
  import WorkPullingPattern._
  
  val modelSpace = new ModelSpace(Vector(new RealMetaParameter("lr",new RealSet(0.1,1.0))), Vector())
  def getRandomConfigs = for (i <- 1 to 100) yield modelSpace.drawRandom
  
  "A random evaluator simulation" should "evaluate concurrently" in {
    val system = ActorSystem("Simulator")
    

    val numWorkers = 50
    val scorerActor = system.actorOf(Props(new ModelScorer(modelSpace, new RandomAcquisitionFunction)), name = "scorer")
    val ev = new MockRandomModelEvaluator
    val master = system.actorOf(Props(new ModelConfigEvaluator[ModelConfig](scorerActor)), name = "master")
    val workers = for (i <- 1 to numWorkers) yield system.actorOf(Props(new ModelConfigEvalWorker(master, scorerActor, ev)), name = "worker"+i)

    workers foreach {w => master ! RegisterWorker(w)}    
    
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
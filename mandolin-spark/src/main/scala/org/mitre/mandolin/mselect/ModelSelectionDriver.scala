package org.mitre.mandolin.mselect

import org.apache.spark.SparkContext
import org.mitre.mandolin.mselect.WorkPullingPattern.{ProvideWork, RegisterWorker}
import org.mitre.mandolin.util.LocalIOAssistant
import akka.actor.{PoisonPill, ActorSystem, Props}

/**
  * Created by jkraunelis on 1/1/17.
  */
object ModelSelectionDriver {

  def main(args: Array[String]): Unit = {
    val system = ActorSystem("Simulator")

    val sc = new SparkContext
    val io = new LocalIOAssistant
    val trainFile = args(0)
    val testFile = args(1)
    val numWorkers = args(2).toInt
    val numThreads = args(3)
    val workerBatchSize = args(4).toInt
    // set up model space
    val lrParam = new RealMetaParameter("lr", new RealSet(0.01, 1.0))
    val methodParam = new CategoricalMetaParameter("method", new CategoricalSet(Vector("adagrad", "sgd")))
    val trainerThreadsParam = new CategoricalMetaParameter("numTrainerThreads", new CategoricalSet(Vector(numThreads)))
    val modelSpace = new ModelSpace(Vector(lrParam), Vector(methodParam, trainerThreadsParam))
    // end model space
    val trainBC = sc.broadcast(io.readLines(trainFile).toVector)
    val testBC = sc.broadcast(io.readLines(testFile).toVector)
    val ev = new SparkModelEvaluator(sc, trainBC, testBC)
    val master = system.actorOf(Props(new ModelConfigEvaluator[ModelConfig]), name = "master")
    val scorerActor = system.actorOf(Props(new ModelScorer(modelSpace, new RandomAcquisitionFunction, master)), name = "scorer")
    val workers = 1 to numWorkers map (i => system.actorOf(Props(new ModelConfigEvalWorker(master, scorerActor, ev, workerBatchSize)), name = "worker" + i))

    workers.foreach(worker => master ! RegisterWorker(worker))
    //master ! ProvideWork(1) // this starts things off
    // master should request work from the scorer if it doesn't have any

    //Thread.sleep(1000 * 60 * 30)

    //workers.foreach(worker => worker ! PoisonPill )
    //scorerActor ! PoisonPill
    //master ! PoisonPill


  }

}

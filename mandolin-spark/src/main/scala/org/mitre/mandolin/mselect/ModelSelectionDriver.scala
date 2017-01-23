package org.mitre.mandolin.mselect

import org.apache.spark.SparkContext
import org.mitre.mandolin.mselect.WorkPullingPattern.RegisterWorker
import org.mitre.mandolin.util.LocalIOAssistant
import akka.actor.{PoisonPill, ActorSystem, Props}
import scala.concurrent.{ExecutionContext }
import java.util.concurrent.Executors
import org.mitre.mandolin.glp.{ GLPTrainerBuilder, GLPModelSettings }

/**
  * Created by jkraunelis on 1/1/17.
  */
object ModelSelectionDriver {

  def main(args: Array[String]): Unit = {
    

    val sc = new SparkContext
    val io = new LocalIOAssistant
    val trainFile = args(0)
    val testFile = args(1)
    val numWorkers = args(2).toInt
    val numThreads = args(3)
    val workerBatchSize = args(4).toInt
    
    implicit val ec = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numWorkers))
    val system = ActorSystem("Simulator")
    
    val settings = (new GLPModelSettings).withSets(Seq(
        ("mandolin.trainer.train-file", trainFile),
        ("mandolin.trainer.test-file", testFile)
        ))
        
    val (trainer, nn) = GLPTrainerBuilder(settings)
    

    val featureExtractor = trainer.getFe
    featureExtractor.getAlphabet.ensureFixed // fix the alphabet
    
    // set up model space
    val lrParam = new RealMetaParameter("lr", new RealSet(0.01, 1.0))
    val methodParam = new CategoricalMetaParameter("method", new CategoricalSet(Vector("adagrad", "sgd")))
    val trainerThreadsParam = new CategoricalMetaParameter("numTrainerThreads", new CategoricalSet(Vector(numThreads)))
    val modelSpace = new ModelSpace(Vector(lrParam), Vector(methodParam, trainerThreadsParam), nn)
    // end model space

    val trVecs = io.readLines(trainFile) map { l => featureExtractor.extractFeatures(l)}    
    val tstVecs = io.readLines(testFile) map { l => featureExtractor.extractFeatures(l)}
    val trainBC = sc.broadcast(trVecs.toVector)
    val testBC = sc.broadcast(tstVecs.toVector)
    val ev = new SparkModelEvaluator(sc, trainBC, testBC)
    val master = system.actorOf(Props(new ModelConfigEvaluator[ModelConfig]), name = "master")
    val acqFun = new BayesianNNAcquisitionFunction
    val scorerActor = system.actorOf(Props(new ModelScorer(modelSpace, acqFun, master, 2400, 64, 64, 640)), name = "scorer")
    val workers = 1 to numWorkers map (i => system.actorOf(Props(new ModelConfigEvalWorker(master, scorerActor, ev, workerBatchSize)), name = "worker" + i))
    Thread.sleep(4000)
    workers.foreach(worker => master ! RegisterWorker(worker))

    //master ! ProvideWork(1) // this starts things off
    // master should request work from the scorer if it doesn't have any

    //Thread.sleep(1000 * 60 * 30)

    //workers.foreach(worker => worker ! PoisonPill )
    //scorerActor ! PoisonPill
    //master ! PoisonPill


  }

}

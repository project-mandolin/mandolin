package org.mitre.mandolin.predict.spark
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import scala.reflect.ClassTag
import org.mitre.mandolin.optimize.{ Weights, LossGradient, BatchEvaluator, GenData  }
import org.mitre.mandolin.optimize.spark.{DistributedOptimizerEstimator, RDDData}
import org.mitre.mandolin.util.IOAssistant
import org.mitre.mandolin.util.spark.SparkIOAssistant
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.storage.StorageLevel
import org.mitre.mandolin.predict.{Predictor, OutputConstructor, EvalPredictor, Confusion, ConfusionMatrix}


/**
 * Trains a model and applies a predictor to a test
 * set. The predictor result is also provided to an output constructor that creates
 * a string representation for each instance - e.g. the posterior distribution.
 */ 
class TrainDecoder[IType, U: ClassTag, W <: Weights[W]: ClassTag, R: ClassTag](
    val trainer: Trainer[IType, U, W],
  val predictor: Predictor[U, W, R],
  val outputConstructor: OutputConstructor[IType,Seq[(Float,R)],String],
  val totalEpochs: Int,
  sc: SparkContext,
  persistLevel: StorageLevel = StorageLevel.MEMORY_ONLY) {
  
  var elapsedTrainingTime = 0.0
  var append = false
  val evPr = new PosteriorDecoder(trainer.fe, predictor, outputConstructor)
  
  def trainAndDecode(train: RDD[IType], test: RDD[IType]) = {
    val (weights,trainLoss) = trainer.trainWeights(train)    
    weights.decompress() // ensure weights decompressed before evaluation (as there are many map tasks)
    val wBc = sc.broadcast(weights)
    evPr.run(test, wBc)
  }
}

class TrainTester[IType, U: ClassTag, W <: Weights[W]: ClassTag, G <: LossGradient[G], R: ClassTag, C <: Confusion[C]: ClassTag](
  val trainer: Trainer[IType, U, W],
  val predictor: EvalPredictor[U, W, R, C],
  val totalEpochs: Int,
  val evalFrequency: Int,
  sc: SparkContext,
  detailFile: Option[String] = None,
  persistLevel: StorageLevel = StorageLevel.MEMORY_ONLY,
  batchEvaluator: Option[BatchEvaluator[U, W, G]] = None) {
  
  val evPr = new EvalDecoder(trainer.fe, predictor)
  var elapsedTrainingTime = 0.0
  var append = false
  val io = new SparkIOAssistant(sc)
  
  def logDetails(trainingLoss: Double, testAccuracy: Double, testAuROC: Double, accAt50: Double, 
      accAt30: Double, elapsedTime: Double, epoch: Int) = {
    detailFile map {f =>
      val writer = io.getPrintWriterFor(f, append)
      if (!append) {
        writer.write("Epoch" + "\t" + "ElapsedTime" + "\t" + "TrainingLoss" + "\t" + "TestAccuracy" + "\t" + "TestAUROC" + "\t" + "Acc@50" + "\t" + "Acc@30" + "\n")
      }
      writer.write(epoch.toString + "\t" + elapsedTime + "\t" + trainingLoss + "\t" + testAccuracy + "\t" + testAuROC + "\t" + accAt50 + "\t" + accAt30 + "\n")      
      writer.close()
      append = true
      }
  }
  
  def trainAndTest(train: RDD[IType], test: RDD[IType]) = {
    val trainVectors = trainer.extractFeatures(train).persist(persistLevel)
    val testVectors  = evPr.extractFeatures(test).persist(persistLevel)
    val totalNumTraining = trainVectors.count() // trigger rdd collection/storage to get better timing information
    for (i <- 1 to totalEpochs / evalFrequency) {      
      val t = System.nanoTime()
      val (weights,trainLoss) = trainer.retrainWeights(trainVectors, evalFrequency)
      elapsedTrainingTime += ((System.nanoTime - t) / 1E9) // just compute training wall time here
      weights.decompress() // ensure weights decompressed before evaluation (as there are many map tasks)
      val wBc = sc.broadcast(weights)      
      val confusion = evPr.evalUnits(testVectors, wBc)
      val confMat = confusion.getMatrix
      val acc = confMat.getAccuracy
      val auRoc = if (confMat.dim == 2) confusion.getAreaUnderROC(1) else confusion.getTotalAreaUnderROC()
      val accAt50 = confusion.getAccuracyAtThroughPut(0.5)
      val accAt30 = confusion.getAccuracyAtThroughPut(0.3)
      val reportedTrainLoss = batchEvaluator match {
        case None => trainLoss
        case Some(ev) => ev.evaluate(new RDDData(trainVectors) : GenData[U], weights).loss
      } 
      logDetails(reportedTrainLoss, acc, auRoc, accAt50, accAt30, elapsedTrainingTime, (i * evalFrequency))      
    }
    trainer.retrainWeights(trainVectors, 1)
  }
}

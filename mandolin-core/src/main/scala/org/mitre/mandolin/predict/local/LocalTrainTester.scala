package org.mitre.mandolin.predict.local
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import scala.reflect.ClassTag

import org.mitre.mandolin.optimize.{ Weights, LossGradient, Updater }
import org.mitre.mandolin.predict.{Confusion, EvalPredictor, Predictor, OutputConstructor}


class LocalTrainDecoder[IType, U: ClassTag, W <: Weights[W]: ClassTag, R: ClassTag](
    val trainer: LocalTrainer[IType, U, W],
  val predictor: Predictor[U, W, R],
  val outputConstructor: OutputConstructor[IType,Seq[(Float,R)],String],
  val totalEpochs: Int) {
  
  var elapsedTrainingTime = 0.0
  var append = false
  val evPr = new LocalPosteriorDecoder(trainer.fe, predictor, outputConstructor)
  
  def trainAndDecode(train: Vector[IType], test: Vector[IType]) = {
    val (weights,trainLoss) = trainer.trainWeights(train)    
    weights.decompress() // ensure weights decompressed before evaluation (as there are many map tasks)
    evPr.run(test, weights)
  }
}

class LocalTrainTester[IType, U: ClassTag, W <: Weights[W]: ClassTag, R: ClassTag, C <: Confusion[C]: ClassTag](
  val trainer: LocalTrainer[IType, U, W],
  val predictor: EvalPredictor[U, W, R, C],
  val totalEpochs: Int,
  val evalFrequency: Int,  
  detailFile: Option[String] = None) {
  
  val evPr = new LocalEvalDecoder(trainer.fe, predictor)
  var elapsedTrainingTime = 0.0
  var append = false
  
  def logDetails(trainingLoss: Double, testAccuracy: Double, testAuROC: Double, accAt50: Double, 
      accAt30: Double, elapsedTime: Double, epoch: Int) = {
    detailFile map {f =>
      val writer = new java.io.FileWriter(f, append) // append
      if (!append) {
        writer.write("Epoch" + "\t" + "ElapsedTime" + "\t" + "TrainingLoss" + "\t" + "TestAccuracy" + "\t" + "TestAUROC" + "\t" + "Acc@50" + "\t" + "Acc@30" + "\n")
      }
      writer.write(epoch.toString + "\t" + elapsedTime + "\t" + trainingLoss + "\t" + testAccuracy + "\t" + testAuROC + "\t" + accAt50 + "\t" + accAt30 + "\n")      
      writer.close()
      append = true
      }
  }
  
  def trainAndTest(train: Vector[IType], test: Vector[IType]) = {
    val trainVectors = trainer.extractFeatures(train)
    val testVectors  = evPr.extractFeatures(test)
    val totalNumTraining = trainVectors.length // trigger rdd collection/storage to get better timing information
    for (i <- 1 to totalEpochs / evalFrequency) {      
      val t = System.nanoTime()
      val (weights,trainLoss) = trainer.retrainWeights(trainVectors, evalFrequency)
      elapsedTrainingTime += ((System.nanoTime - t) / 1E9) // just compute training wall time here
      
      val confusion = evPr.evalUnits(testVectors, weights)      
      val confMat = confusion.getMatrix
      val acc = confMat.getAccuracy
      val auRoc = if (confMat.dim == 2) confusion.getAreaUnderROC(1) else confusion.getTotalAreaUnderROC()
      val accAt50 = confusion.getAccuracyAtThroughPut(0.5)
      val accAt30 = confusion.getAccuracyAtThroughPut(0.3)
      logDetails(trainLoss, acc, auRoc, accAt50, accAt30, elapsedTrainingTime, (i * evalFrequency))      
    }
    trainer.retrainWeights(trainVectors, 1)
  }
}

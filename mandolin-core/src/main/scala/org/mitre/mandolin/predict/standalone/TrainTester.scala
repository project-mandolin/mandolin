package org.mitre.mandolin.predict.standalone

/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import scala.reflect.ClassTag

import org.mitre.mandolin.optimize.Weights
import org.mitre.mandolin.predict.{Confusion, EvalPredictor, Predictor, OutputConstructor, ConfusionMatrix}

class TrainDecoder[IType, U: ClassTag, W <: Weights[W] : ClassTag, R: ClassTag](
                                                                                      val trainer: Trainer[IType, U, W],
                                                                                      val predictor: Predictor[U, W, R],
                                                                                      val outputConstructor: OutputConstructor[IType, Seq[(Float, R)], String],
                                                                                      val totalEpochs: Int) {

  var elapsedTrainingTime = 0.0
  var append = false
  val evPr = new PosteriorDecoder(trainer.getFe, predictor, outputConstructor)

  def trainAndDecode(train: Vector[IType], test: Vector[IType]) = {
    val (weights, trainLoss) = trainer.trainWeights(train)
    weights.decompress() // ensure weights decompressed before evaluation (as there are many map tasks)
    evPr.run(test, weights)
  }
}

class TrainTester[IType, U: ClassTag, W <: Weights[W] : ClassTag, R: ClassTag, C <: Confusion[C] : ClassTag](
                                                                                                                   val trainer: Trainer[IType, U, W],
                                                                                                                   val predictor: EvalPredictor[U, W, R, C],
                                                                                                                   val totalEpochs: Int,
                                                                                                                   val evalFrequency: Int,
                                                                                                                   detailFile: Option[String] = None) {
  val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  
  var elapsedTrainingTime = 0.0
  var append = false

  def logDetails(trainingLoss: Double, testAccuracy: Double, testAuROC: Double, accAt50: Double,
                 accAt30: Double, elapsedTime: Double, epoch: Int) = {
    detailFile map { f =>
      val writer = new java.io.FileWriter(f, append) // append
      if (!append) {
        writer.write("Epoch" + "\t" + "ElapsedTime" + "\t" + "TrainingLoss" + "\t" + "TestAccuracy" + "\t" + "TestAUROC" + "\t" + "Acc@50" + "\t" + "Acc@30" + "\n")
      }
      writer.write(epoch.toString + "\t" + elapsedTime + "\t" + trainingLoss + "\t" + testAccuracy + "\t" + testAuROC + "\t" + accAt50 + "\t" + accAt30 + "\n")
      writer.close()
      append = true
    }
  }

  /**
    * Trains a model and evaluates/tests it on test data periodically. Evaluations are logged.
    */
  def trainAndTest(train: Vector[IType], test: Vector[IType]) = {
    val evPr = new EvalDecoder(trainer.getFe, predictor)
    val trainVectors = trainer.extractFeatures(train)
    val testVectors = evPr.extractFeatures(test)
    val totalNumTraining = trainVectors.length // trigger rdd collection/storage to get better timing information
    for (i <- 1 to totalEpochs / evalFrequency) {
      val t = System.nanoTime()
      val (weights, trainLoss) = trainer.retrainWeights(trainVectors, evalFrequency)
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

  def crossValidate(train: Vector[U], folds: Int = 5) : ConfusionMatrix = {
    logger.info("Initiating cross validation using " + folds + " folds")
    val numPoints = train.length
    val subparts = train.grouped(numPoints / folds).toVector
    // train and evaluate on each fold
    val t0 = System.nanoTime
    val evPr = new NonExtractingEvalDecoder(predictor)    
    val ss = for (i <- 0 until folds) yield {
      val foldTrainer = trainer.copy() // copy the entire trainer to ensure weights aren't shared across folds
      val tstSet = subparts(i)
      val trSet = (subparts.zipWithIndex).foldLeft(Vector(): Vector[U]){case (ac, (v,k)) => if (k == i) ac else ac ++ v}
      val (weights,_) = foldTrainer.retrainWeights(trSet, totalEpochs)
      val r = evPr.evalUnits(tstSet, weights)
      logger.info("Finished training for fold " + i + " with accuracy: " + r.getMatrix.getAccuracy)
      // make sure to reset weights so we don't train next fold starting from current one
      // other model types should probably re-instantiate a trainer object for each fold      
      r
    }
    val finalConf = ss.reduce {_ compose _}
    val m = finalConf.getMatrix
    val acc = m.getAccuracy
    val auRoc = if (m.dim == 2) finalConf.getAreaUnderROC(1) else finalConf.getTotalAreaUnderROC()
    val accAt50 = finalConf.getAccuracyAtThroughPut(0.5)
    val accAt30 = finalConf.getAccuracyAtThroughPut(0.3)
    logDetails(0.0, acc, auRoc, accAt50, accAt30, ((System.nanoTime - t0) / 1E9), 1)    
    m
  }

  /**
    * Takes a train/test split, builds a model on the train set and evaluates on the test
    * set. By default the score is 1 - auROC.
    */
  def extractTrainAndScore(train: Vector[IType], test: Vector[IType]): Double = {
    val evPr = new EvalDecoder(trainer.getFe, predictor)
    val trVecs = trainer.extractFeatures(train)
    val testVecs = evPr.extractFeatures(test)
    trainAndScore(trVecs, testVecs)
  }

  def trainAndScore(train: Vector[U], test: Vector[U]): Double = {
    val evPr = new EvalDecoder(trainer.getFe, predictor)
    val (weights, trainLoss) = trainer.retrainWeights(train, totalEpochs)
    val confusion = evPr.evalUnits(test, weights)
    val confMat = confusion.getMatrix
    val auRoc = if (confMat.dim == 2) confusion.getAreaUnderROC(1) else confusion.getTotalAreaUnderROC()
    1.0 - auRoc
  }
}

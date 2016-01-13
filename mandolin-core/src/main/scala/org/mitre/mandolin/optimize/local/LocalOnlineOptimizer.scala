package org.mitre.mandolin.optimize.local
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.optimize.{Weights, LossGradient,Updater,TrainingUnitEvaluator,ModelWriter,OptimizerWriter,EpochProcessor}
import org.mitre.mandolin.config.{ LearnerSettings, OnlineLearnerSettings }
import scala.reflect.ClassTag

trait LocalOptimizerEstimator[T, W <: Weights[W]] {
  def estimate(data: Vector[T], mxEpochs: Option[Int] = None): (W, Double)
}

/**
 * Provides the functionality of the distributed OnlineOptimizer
 * without using Spark/RDDs to provide non-Spark training that supports
 * the same classifier API.
 * 
 * @author wellner
 */
class LocalOnlineOptimizer[T, W <: Weights[W]: ClassTag, LG <: LossGradient[LG], U <: Updater[W, LG, U]: ClassTag](
  val initialWeights: W,
  val evaluator: TrainingUnitEvaluator[T, W, LG, U],
  val initialUpdater: U,
  maxEpochs: Int,
  numSubEpochs: Int,
  val workersPerPartition: Int,
  optimizationDetails: Option[String],
  modelWriterOpt: Option[ModelWriter[W]] = None,
  synchronous: Boolean = false,
  skipProb: Double = 0.0,
  miniBatchSize : Int = 1,
  shuffle: Boolean = true) extends LocalOptimizerEstimator[T, W] {
  
  def this(_iw: W, _ev: TrainingUnitEvaluator[T,W,LG, U], _u: U, _as: OnlineLearnerSettings) = {
    this(_iw, _ev, _u, _as.numEpochs, _as.numSubEpochs,
          _as.numThreads, _as.detailsFile,
          synchronous = _as.synchronous, skipProb = _as.skipProb, 
          miniBatchSize = _as.miniBatchSize, shuffle = true)
  }
  
  var numEpochs = 0

  val optOut = new OptimizerWriter(optimizationDetails)

  protected var weights = initialWeights
  protected var updater = initialUpdater

  def estimate(data: Vector[T], mxEpochs: Option[Int] = None): (W, Double) = {
    val numPartitions = 1
    val t0 = System.nanoTime()
    val mx = mxEpochs.getOrElse(maxEpochs)
    var finalLoss = 0.0
    for (i <- 1 to mx) {
      val shdata = if (shuffle) util.Random.shuffle(data) else data
      val (loss, newWeights, newUpdater) = processEpoch(shdata, numPartitions, i, weights, updater)
      val ct = (System.nanoTime() - t0) / 1.0E9
      optOut.writeln(i.toString() + "\t" + loss + "\t" + ct)
      weights = newWeights
      updater = newUpdater
      finalLoss = loss
      modelWriterOpt foreach { w => w.writeModel(weights) }
    }
    (weights, finalLoss)
  }

  def processEpoch(data: Vector[T], numPartitions: Int, curEpoch: Int, currentWeights: W, currentUpdater: U): (Double, W, U) = {
    val ep = new EpochProcessor[T, W, LG, U](evaluator, workersPerPartition = workersPerPartition, 
        synchronous = synchronous, numSubEpochs = numSubEpochs, skipProb = skipProb)
    val ss = System.nanoTime
    val loss = ep.processPartitionWithinEpoch(curEpoch, data, currentWeights, currentUpdater)
    (loss, currentWeights, currentUpdater)
  }
  
}

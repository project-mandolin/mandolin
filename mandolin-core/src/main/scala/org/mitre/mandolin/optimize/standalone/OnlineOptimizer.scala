package org.mitre.mandolin.optimize.standalone
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.mlp.MandolinMLPSettings
import org.mitre.mandolin.optimize.{EpochProcessor, LossGradient, ModelWriter, OptimizerWriter, TrainingUnitEvaluator, Updater, Weights}
import org.slf4j.LoggerFactory
import scala.reflect.ClassTag

trait OptimizerEstimator[T, W <: Weights[W]] {

  def copy() : OptimizerEstimator[T, W]

  def estimate(data: Vector[T], mxEpochs: Option[Int] = None): (W, Double)
}

/**
 * Provides the functionality of the distributed OnlineOptimizer
 * without using Spark/RDDs to provide non-Spark training that supports
 * the same classifier API.
 *
 * @author wellner
 */
class OnlineOptimizer[T, W <: Weights[W]: ClassTag, LG <: LossGradient[LG], U <: Updater[W, LG, U]: ClassTag](
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
  shuffle: Boolean = true) extends OptimizerEstimator[T, W] {
  
  def this(_iw: W, _ev: TrainingUnitEvaluator[T,W,LG, U], _u: U, _as: MandolinMLPSettings) = {
    this(_iw, _ev, _u, _as.numEpochs, _as.numSubEpochs,
          _as.numThreads, _as.detailsFile,
          synchronous = _as.synchronous, skipProb = _as.skipProb, 
          miniBatchSize = _as.miniBatchSize, shuffle = true)
  }
  
  val logger = LoggerFactory.getLogger(this.getClass)
  var numEpochs = 0

  val optOut = new OptimizerWriter(optimizationDetails)

  protected var weights = initialWeights
  protected var updater = initialUpdater
  
  def copy() : OptimizerEstimator[T, W] = {
    new OnlineOptimizer(initialWeights.copy(), evaluator.copy(), initialUpdater.copy(), 
        maxEpochs, numSubEpochs, workersPerPartition, optimizationDetails, modelWriterOpt, synchronous, skipProb, miniBatchSize, shuffle)
  }
  
  def estimate(data: Vector[T], mxEpochs: Option[Int] = None): (W, Double) = {
    val numPartitions = 1
    val t0 = System.nanoTime()
    val mx = mxEpochs.getOrElse(maxEpochs)
    var finalLoss = 0.0
    var lastTime = System.nanoTime
    for (i <- 1 to mx) {
      val shdata = if (shuffle) util.Random.shuffle(data) else data
      val (loss, time, newWeights, newUpdater) = processEpoch(shdata, numPartitions, i, weights, updater)
      val curTime = System.nanoTime()
      val ct = ( curTime - t0) / 1.0E9
      optOut.writeln(i.toString() + "\t" + loss + "\t" + ct)
      weights = newWeights
      updater = newUpdater
      finalLoss = loss
      // logger.info("Training epoch " + i + " completed. Time = " + ((curTime - lastTime) / 1.0E9))
      lastTime = curTime
      modelWriterOpt foreach { w => w.writeModel(weights) }
    }
    (weights, finalLoss)
  }

  def processEpoch(data: Vector[T], numPartitions: Int, curEpoch: Int, currentWeights: W, currentUpdater: U): (Double, Long, W, U) = {
    val ep = new EpochProcessor[T, W, LG, U](evaluator, workersPerPartition = workersPerPartition, 
        synchronous = synchronous, numSubEpochs = numSubEpochs, skipProb = skipProb)
    val ss = System.nanoTime
    val (loss,time) = ep.processPartitionWithinEpoch(curEpoch, data, currentWeights, currentUpdater, 0L)
    (loss, time, currentWeights, currentUpdater)
  }
  
}

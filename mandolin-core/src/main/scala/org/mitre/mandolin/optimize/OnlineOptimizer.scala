package org.mitre.mandolin.optimize
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import java.util.concurrent.locks.ReentrantReadWriteLock
import scala.reflect.ClassTag
import org.mitre.mandolin.util.{ Tensor1, DenseTensor1 }
import org.mitre.mandolin.config.{ LearnerSettings, OnlineLearnerSettings }



/**
 * Handles the processing of a single partition within a single <i>epoch</i>. The partition is processed by spawning separate
 * within-process worker threads (using Scala's parallel collections).  Each thread processes a portion of the training data
 * within the partition and updates the model parameters local to this worker/partition, possibly asynchronously.
 * @param evaluator Evaluator that evaluates loss and gradient for a single datapoint or mini-batch
 * @param workersPerPartition Number of threads on each node to compute gradient-based updates
 * @param synchronous Set to true to prevent asynchronous "Hogwild" updates
 * @param numSubEpochs Number of passes over the data on each partition during a map-reduce epoch (default 1)
 * @param skipProb Probability of skipping a data point (to achieve fast undersampling)
 * @param miniBatchSize Number of datapoints to compute loss/gradient for each update
 * @author Ben Wellner, Karthik Prakhya
 */
class EpochProcessor[T, W <: Weights[W], LG <: LossGradient[LG], U <: Updater[W, LG, U]](evaluator: TrainingUnitEvaluator[T, W, LG],
  val workersPerPartition: Int = 16, synchronous: Boolean = false, numSubEpochs: Int = 1, skipProb: Double = 0.0, 
  miniBatchSize: Int = 1, concurrentBatch: Int = 0)
  extends Serializable {

  /*
   * Get a new thread processor for this epoch; either asynchronous or synchronous
   */
  def getThreadProcessor(data: Vector[T], w: W, updater: U, rwLock: Option[ReentrantReadWriteLock]) = rwLock match {
    case Some(rwL) => new SynchronousThreadProcessor(data, w, evaluator.copy(), updater, rwL, skipProb, miniBatchSize)
    case None => new AsynchronousThreadProcessor(data, w, evaluator.copy(), updater, skipProb, miniBatchSize)
  }
  
  // this may better handle very large datasets
  def processPartitionsConcurrently(data: Iterator[T], w: W, updater: U) : Double = {
    if (concurrentBatch > 0)      
      data.grouped(concurrentBatch).foldLeft(0.0){case (ac, seq) =>
        ac + processPartitionWithinEpoch(0,seq.toVector, w, updater) }
    else
      processPartitionWithinEpoch(0,data.toVector,w,updater)
  }

  def processPartitionWithinEpoch(curEpoch: Int, partitionInsts: Vector[T], w: W, updater: U): Double = {
    val factor = (partitionInsts.size.toDouble / workersPerPartition).ceil.toInt
    val subPartitions = partitionInsts.grouped(factor) // split up data into sub-slices
    val rwLock = if (synchronous) Some(new ReentrantReadWriteLock) else None
    val workers = (subPartitions map { sub => getThreadProcessor(sub, w, updater, rwLock) }).toList.par
    var totalLoss: Double = 0
    for (i <- 1 to numSubEpochs) {
      val subLoss = workers map { _.process() } reduce { _ + _ }
      totalLoss += subLoss
    }
    totalLoss
  }
}

abstract class AbstractThreadProcessor[T] {
  def process(): Double
}

/**
 * Implements "Hogwild" asynchronous updates of the local copy of the weights.
 */
class AsynchronousThreadProcessor[T, W <: Weights[W], LG <: LossGradient[LG], U <: Updater[W, LG, U]](
  val data: Vector[T],
  val weights: W,
  val evaluator: TrainingUnitEvaluator[T, W, LG],
  val updater: U,
  skipProb: Double = 0.0,
  miniBatchSize: Int = 1)
  extends AbstractThreadProcessor[T] {
  val n = data.length
  def process() = {
    @volatile var totalLoss = 0.0
    var continue = true
    var i = 0; while (continue) {
      if ((skipProb <= 0.0) || (util.Random.nextDouble() < skipProb)) {
        val totalLossGrad = if (miniBatchSize > 1) {
          val items = math.min(miniBatchSize + i, n) - i
          val lossGrads = for (jj <- i until math.min(miniBatchSize + i, n)) yield evaluator.evaluateTrainingUnit(data(jj), weights)
          i += items
          updater.resetMass(1.0f / items.toFloat) // set the update mass in updater to the number if 1.0/items to take average of gradient
          lossGrads reduce { _ ++ _ }
        } else {
          val r = evaluator.evaluateTrainingUnit(data(i), weights)
          i += 1
          r
        }
        updater.updateWeights(totalLossGrad, weights)
        totalLoss += totalLossGrad.loss
      } else i += 1
      if (i >= n) continue = false
    }
    totalLoss
  }
}

/**
 * Simple synchronous updater that uses read-write locks to ensure read weights are modified
 * during process of reading their values
 *
 */
class SynchronousThreadProcessor[T, W <: Weights[W], LG <: LossGradient[LG], U <: Updater[W, LG, U]](
  val data: Vector[T],
  val weights: W,
  val evaluator: TrainingUnitEvaluator[T, W, LG],
  val updater: U,
  rwLock: ReentrantReadWriteLock,
  skipProb: Double = 0.0,
  miniBatchSize: Int = 1) extends AbstractThreadProcessor[T] {
  val readLock = rwLock.readLock()
  val writeLock = rwLock.writeLock()
  val n = data.length

  def process() = {
    var totalLoss = 0.0
    var continue = true
    var i = 0; while (continue) {
      if ((skipProb <= 0.0) || (util.Random.nextDouble() < skipProb)) {
        readLock.lock()
        val totalLossGrad = if (miniBatchSize > 1) {
          val items = math.min(miniBatchSize + i, n) - i
          val lossGrads = for (jj <- i until math.min(miniBatchSize + i, n)) yield evaluator.evaluateTrainingUnit(data(jj), weights)
          i += items
          lossGrads reduce { _ ++ _ }
        } else {
          val r = evaluator.evaluateTrainingUnit(data(i), weights)
          i += 1
          r
        }
        readLock.unlock()
        writeLock.lock()
        updater.updateWeights(totalLossGrad, weights)
        totalLoss += totalLossGrad.loss
        writeLock.unlock()
      } else i += 1
      if (i >= n) continue = false
    }
    totalLoss
  }
}

/*
 * Provides functionality/callbacks to write out intermediate loss values
 * and other data after each epoch
 * @param filePath A filepath (option) that specifies where post-epoch data should be written
 */
class OptimizerWriter(filePath: Option[String]) {
  private val f = filePath map { ss => new java.io.File(ss) }
  def writeln(s: String) = {
    val writer = f map { fi => new java.io.FileWriter(fi, true) }
    writer foreach { w => w.write(s); w.write('\n'); w.close() }
  }
  def write(s: String) = {
    val writer = f map { fi => new java.io.FileWriter(fi, true) }
    writer foreach { os => os.write(s); os.close() }
  }
}

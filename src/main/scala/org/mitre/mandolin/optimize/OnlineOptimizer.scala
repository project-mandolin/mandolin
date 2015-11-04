package org.mitre.mandolin.optimize
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import java.util.concurrent.locks.ReentrantReadWriteLock
import scala.reflect.ClassTag
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.SparkContext
import org.mitre.mandolin.util.{ Tensor1, DenseTensor1 }
import org.mitre.mandolin.config.{ LearnerSettings, OnlineLearnerSettings }
import org.apache.spark.Partitioner

/**
 * RDD Partitioner which guarantees that RDD elements are distributed amongst
 * partitions randomly.
 */
class RandomPartitioner(partitions: Int) extends Partitioner {

  def getPartition(key: Any): Int = {
    val r = new scala.util.Random()
    r.nextInt(partitions)

  }

  def numPartitions(): Int = {
    return partitions
  }
}

/**
 * This class drives online optimization.  It requires three type-parameterizable components:
 * 1) a set of model weights that subclasses Weights, 2) a training evaluator that provides a LossGradient
 * for a given training unit, a subclass of TrainingUnitEvaluator and 3) an Updater object that modifies
 * the parameters for a given LossGradient.
 *
 * The scheme works by localizing a single partition (or small number of partitions) to each worker node.
 * Each partition is processed by an EpochProcessor object.  The EpochProcessor returns the model parameters obtained
 * via online updates for only that partition (data parallelism).  The reduce step involves composition the
 * resulting model parameters from each EpochProcessor - typically, this is done by simply averaging the parameters.
 * Other composition operators can be defined.
 * @param sc Spark context
 * @param initialWeights initialized weights passed in to optimizer
 * @param evaluator Evaluate loss function with current weights
 * @param initialUpdater Weight updater (e.g. sgd, adagrad, ..), initialized as appropriate
 * @param maxEpochs maximum number of map-reduce epochs to perform
 * @param numSubEpochs number of passes over local data per map-reduce iteration
 * @param workersPerPartition number of threads operating over data per partition (usually 1 partition per node)
 * @param optimizationDetails output file where progress of optimizer is written to
 * @param synchronous force synchronous parameter updates on each node (i.e. do NOT use "Hogwild" training)
 * @param ensureSparse pass to weights and updater to request compression (e.g. if weights are sparse)
 * @param skipProb skip a data instance with probability skipProb
 * @param miniBatchSize use mini-batches of this size (defaults to 1)
 * @param oversample ratio to under/oversample data to each partition on each map-reduce epoch - a value of 0.0 means NO sampling whatsoever
 *
 * @author Ben Wellner, Karthik Prakhya
 */
class OnlineOptimizer[T: ClassTag, W <: Weights[W]: ClassTag, LG <: LossGradient[LG], U <: Updater[W, LG, U]: ClassTag](
  sc: SparkContext,
  val initialWeights: W,
  val evaluator: TrainingUnitEvaluator[T, W, LG],
  val initialUpdater: U,
  maxEpochs: Int,
  numSubEpochs: Int,
  val workersPerPartition: Int,
  optimizationDetails: Option[String],
  synchronous: Boolean = false,
  ensureSparse: Boolean = true,
  skipProb: Double = 0.0,
  miniBatchSize: Int = 1,
  oversample: Double = 0.0) extends DistributedOptimizerEstimator[T, W] {

  def this(_sc: SparkContext, _iw: W, _e: TrainingUnitEvaluator[T, W, LG], _u: U, _as: OnlineLearnerSettings) = {
    this(_sc, _iw, _e, _u, _as.numEpochs, _as.numSubEpochs,
      _as.numThreads, _as.detailsFile,
      synchronous = _as.synchronous, skipProb = _as.skipProb, miniBatchSize = _as.miniBatchSize,
      oversample = _as.oversampleRatio)
  }

  var numEpochs = 0

  val optOut = new OptimizerWriter(optimizationDetails)

  protected var weights = initialWeights
  protected var updater = initialUpdater

  /**
   * Estimates/trains model parameters
   * @param rdd - training data in RDD
   * @param mxEpochs - optional number of training passes
   */
  def estimate(rdd: RDD[T], mxEpochs: Option[Int] = None): (W, Double) = {
    val numPartitions = rdd.partitions.length
    val t0 = System.nanoTime()
    val mx = mxEpochs.getOrElse(maxEpochs)
    var finalLoss = 0.0
    for (i <- 1 to mx) {
      val (loss, newWeights, newUpdater) = processEpoch(rdd, numPartitions, i, weights, updater)
      val ct = (System.nanoTime() - t0) / 1.0E9
      optOut.writeln(i.toString() + "\t" + loss + "\t" + ct)
      newWeights.resetMass()
      newUpdater.resetMass() // need to reset the updater and weights "mass" values for reduce-based averaging
      weights = newWeights
      updater = newUpdater
      finalLoss = loss
    }
    (weights, finalLoss)
  }

  /**
   * @param rdd All data elements represented as an RDD
   * @param numPartitions Number of partitions to create for training
   * @param curEpoch the current epoch/iteration
   * @param currentWeights current set of model weights/parameters
   * @param currentUpdater the current updater to update weights within online training
   * @author wellner
   */
  def processEpoch(rdd: RDD[T], numPartitions: Int, curEpoch: Int, currentWeights: W, currentUpdater: U): (Double, W, U) = {
    if (ensureSparse) {
      currentWeights.compress()
      currentUpdater.compress()
    }
    currentWeights.resetMass(1.0) // ensure the masses on weights are reset
    val bc = sc.broadcast(currentWeights)
    val bcUpdater = sc.broadcast(currentUpdater)
    val ep = new EpochProcessor[T, W, LG, U](evaluator, workersPerPartition = workersPerPartition,
      synchronous = synchronous, numSubEpochs = numSubEpochs, skipProb = skipProb)
    val _ensureSparse = ensureSparse

    val trainRDD =
      if (oversample > 0.0) {
        val withReplacement = (oversample > 1.0)
        val numPartitions = rdd.partitions.length
        val sampled = rdd.sample(withReplacement, oversample).map(x => (x, 1))
        sampled.partitionBy(new RandomPartitioner(numPartitions)).map(x => x._1)
      } else rdd
    val lossAndParamSums = {
      val parameterSet = trainRDD.mapPartitions({ insts =>
        val w = bc.value
        val u = bcUpdater.value
        w.resetMass(1.0)
        u.resetMass(1.0)
        val partitionInsts = insts.toVector // pull in data points to a vector -- note this duplicates between here and local cache
        w.decompress()
        u.decompress()
        val loss = ep.processPartitionWithinEpoch(curEpoch, partitionInsts, w, u)
        if (_ensureSparse) {
          w.compress()
          u.compress()
        }
        Seq((loss, w, u)).toIterator
      }, true)
      parameterSet treeReduce {
        case (l, r) =>
          ((l._1 + r._1), l._2 compose r._2, l._3 compose r._3)
      }
    }
    (lossAndParamSums._1, lossAndParamSums._2, lossAndParamSums._3)
  }
}

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
  val workersPerPartition: Int = 16, synchronous: Boolean = false, numSubEpochs: Int = 1, skipProb: Double = 0.0, miniBatchSize: Int = 1)
  extends Serializable {

  /*
   * Get a new thread processor for this epoch; either asynchronous or synchronous
   */
  def getThreadProcessor(data: Vector[T], w: W, updater: U, rwLock: Option[ReentrantReadWriteLock]) = rwLock match {
    case Some(rwL) => new SynchronousThreadProcessor(data, w, evaluator.copy(), updater, rwL, skipProb, miniBatchSize)
    case None => new AsynchronousThreadProcessor(data, w, evaluator.copy(), updater, skipProb, miniBatchSize)
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
          updater.resetMass(1.0 / items.toDouble) // set the update mass in updater to the number if 1.0/items to take average of gradient
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

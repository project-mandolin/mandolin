package org.mitre.mandolin.optimize.spark

import scala.reflect.ClassTag
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.Partitioner
import org.mitre.mandolin.mlp.MandolinMLPSettings
import org.mitre.mandolin.optimize._

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
 * @param numSubEpochs number of passes over standalone data per map-reduce iteration
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
class DistributedOnlineOptimizer[T: ClassTag, W <: Weights[W]: ClassTag, LG <: LossGradient[LG], U <: Updater[W, LG, U]: ClassTag](
  sc: SparkContext,
  val initialWeights: W,
  val evaluator: TrainingUnitEvaluator[T, W, LG, U],
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

  def this(_sc: SparkContext, _iw: W, _e: TrainingUnitEvaluator[T, W, LG, U], _u: U, _as: MandolinMLPSettings) = {
    this(_sc, _iw, _e, _u, _as.numEpochs, _as.numSubEpochs,
      _as.numThreads, _as.detailsFile,
      synchronous = _as.synchronous, skipProb = _as.skipProb, miniBatchSize = _as.miniBatchSize,
      oversample = _as.oversampleRatio)
  }

  val optOut = new OptimizerWriter(optimizationDetails)

  var numEpochs = 0 // keep some state here as this can be called repeatedly/incrementally
  protected var weights = initialWeights
  protected var updater = initialUpdater
  
  var expectedTime = 0L
  
  def getExpectedEpochTime = expectedTime
  
  def getNewExpectedTime(recentTime: Long) : Long = {
    ((recentTime + (expectedTime / math.sqrt(numEpochs.toDouble))) / (1.0 + (1.0 / math.sqrt(numEpochs.toDouble)))).toLong  
  }     
  
  /**
   * Estimates/trains model parameters
   * @param rdd - training data in RDD
   * @param mxEpochs - optional number of training passes
   */
  def estimate(rdd: RDD[T], mxEpochs: Option[Int] = None): (W, Double) = {
    numEpochs += 1
    val numPartitions = rdd.partitions.length
    val t0 = System.nanoTime()
    val mx = mxEpochs.getOrElse(maxEpochs)
    var finalLoss = 0.0
    for (i <- 1 to mx) {
      val (loss, time, newWeights, newUpdater) = processEpoch(rdd, numPartitions, i, weights, updater, 0L)
      val ct = (System.nanoTime() - t0) / 1.0E9
      optOut.writeln(i.toString() + "\t" + loss + "\t" + ct)
      newWeights.resetMass()
      newUpdater.resetMass() // need to reset the updater and weights "mass" values for reduce-based averaging
      weights = newWeights
      updater = newUpdater
      finalLoss = loss
      // val localExpectedTime = time / numPartitions
      // if (numEpochs < 2) expectedTime = localExpectedTime * 10 // set the previous expected time to large factor more than first epoch expected time to be conservative
      // expectedTime = getNewExpectedTime(localExpectedTime)
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
  def processEpoch(rdd: RDD[T], numPartitions: Int, curEpoch: Int, currentWeights: W, currentUpdater: U, expectedTime: Long): (Double, Long, W, U) = {
    if (ensureSparse) {
      currentWeights.compress()
      currentUpdater.compress()
    }
    currentWeights.resetMass(1.0f) // ensure the masses on weights are reset
    val bc = sc.broadcast(currentWeights)
    val bcUpdater = sc.broadcast(currentUpdater.compress())
    val bcEp = sc.broadcast(new EpochProcessor[T, W, LG, U](evaluator, workersPerPartition = workersPerPartition,
      synchronous = synchronous, numSubEpochs = numSubEpochs, skipProb = skipProb))
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
        val ep = bcEp.value
        w.resetMass(1.0f)
        u.resetMass(1.0f)
        val partitionInsts = util.Random.shuffle(insts.toVector) 
        w.decompress()         
        val ss = System.nanoTime()
        val (loss, time) = ep.processPartitionWithinEpoch(curEpoch, partitionInsts, w, u.decompress(), expectedTime)        
        if (_ensureSparse) {
          w.compress()
        }
        Seq((loss, time, w, u.compress())).toIterator
      }, true)
      parameterSet reduce {
        case (l, r) =>
          ((l._1 + r._1), (l._2 + r._2), l._3 compose r._3, l._4 compose r._4)
      }
    }
    (lossAndParamSums._1, lossAndParamSums._2, lossAndParamSums._3, lossAndParamSums._4)
  }
  
  def processEpochLargeModel1(rdd: RDD[T], numPartitions: Int, curEpoch: Int, currentWeights: W, currentUpdater: U): (Double, W, U) = {
    if (ensureSparse) {
      currentWeights.compress()
      currentUpdater.compress()
    }
    currentWeights.resetMass(1.0f) // ensure the masses on weights are reset
    val bc = sc.broadcast(currentWeights)
    val bcUpdater = sc.broadcast(currentUpdater)
    val bcEp = sc.broadcast(new EpochProcessor[T, W, LG, U](evaluator, workersPerPartition = workersPerPartition,
      synchronous = synchronous, numSubEpochs = numSubEpochs, skipProb = skipProb))
    val _ensureSparse = ensureSparse
    val lossAc = sc.accumulator(0.0)
    val numPartitions = rdd.partitions.length
    val trainRDD =
      if (oversample > 0.0) {
        val withReplacement = (oversample > 1.0)        
        val sampled = rdd.sample(withReplacement, oversample).map(x => (x, 1))
        sampled.partitionBy(new RandomPartitioner(numPartitions)).map(x => x._1)
      } else rdd
    val lossAndParamSums = {
      val parameterSet = trainRDD.mapPartitions({ insts =>
        val w = bc.value
        val u = bcUpdater.value
        val ep = bcEp.value
        w.resetMass(1.0f)
        u.resetMass(1.0f)
        val partitionInsts = insts.toVector // pull in data points to a vector 
        w.decompress()
        u.decompress()
        val (loss,time) = ep.processPartitionWithinEpoch(curEpoch, partitionInsts, w, u, 0L)
        if (_ensureSparse) {
          w.compress()
          u.compress()
        }
        lossAc += loss
        val arW = w.asArray
        val arU = u.asArray  
        // idea here ... 
        // use accumulator to handle loss
        // represent weights and learning rates with tuples (index, weight, learning rate)
        // map index
        //Seq((loss, w, u)).toIterator
        Iterator.tabulate(arW.length)(i => (i, (arW(i), arU(i))))
      }, true)
      parameterSet reduceByKey {(a,b) => ((a._1 + b._1), (a._2 + b._2))}
    }  
    val size = currentWeights.asArray.length
    val wArr = Array.fill(size)(0.0f)
    val uArr = Array.fill(size)(0.0f)
    lossAndParamSums.toLocalIterator foreach { case (ind, (wv, uv)) =>
      wArr(ind) = wv / numPartitions
      uArr(ind) = uv / numPartitions  // need to normalize this back to get the average
    }
    currentWeights.updateFromArray(wArr)
    currentUpdater.updateFromArray(uArr)
    // add capabilities to consturct weights efficiently from indexed iterator 
    (lossAc.value, currentWeights, currentUpdater)
  }
}



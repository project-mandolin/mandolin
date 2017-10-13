package org.mitre.mandolin.embed.spark

import org.apache.spark.{SparkContext}
import org.apache.spark.rdd.RDD
import org.mitre.mandolin.optimize.{TrainingUnitEvaluator, EpochProcessor}
import org.mitre.mandolin.embed._
import scala.reflect.ClassTag

/**
 * @author wellner
 */
class DistributedEmbeddingProcessor[U <: EmbedUpdater[U]: ClassTag](sc: SparkContext,
    val initialWeights: EmbedWeights,
    val evaluator: TrainingUnitEvaluator[SeqInstance, EmbedWeights, EmbedGradient, U],
    val initialUpdater: U,
    workersPerPartition: Int,
    epochs: Int
  ) {
  
  def estimate(rdd: RDD[SeqInstance], mxEpochs: Option[Int] = None): EmbedWeights = {
    val numPartitions = rdd.partitions.length
    val t0 = System.nanoTime()
    var weights = initialWeights
    var updater = initialUpdater
    for (i <- 0 until mxEpochs.getOrElse(epochs)) {
      val (loss, time, newWeights, newUpdater) = processEpoch(rdd, numPartitions, initialWeights, initialUpdater)
      val ct = (System.nanoTime() - t0) / 1.0E9
      newWeights.resetMass()
      newUpdater.resetMass() // need to reset the updater and weights "mass" values for reduce-based averaging
      weights = newWeights
      updater = newUpdater
    }
    weights
  }

def processEpoch(rdd: RDD[SeqInstance], numPartitions: Int, currentWeights: EmbedWeights, currentUpdater: U): (Double, Long, EmbedWeights, U) = {
    currentWeights.compress()
    currentUpdater.compress()
    
    currentWeights.resetMass(1.0f) // ensure the masses on weights are reset
    val bc = sc.broadcast(currentWeights)
    val bcUpdater = sc.broadcast(currentUpdater.compress())
    val bcEp = sc.broadcast(new EpochProcessor[SeqInstance, EmbedWeights, EmbedGradient, U](evaluator, workersPerPartition = workersPerPartition,
      synchronous = false))

    val trainRDD = rdd

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
        val (loss, time) = ep.processPartitionWithinEpoch(1, partitionInsts, w, u.decompress(), 0L)        
        w.compress()   
        // val wArrays = w.embW.
        // XXX - we should return a broken up model here and then reassemble . . . .
        Seq((loss, time, w, u.compress())).toIterator
      }, true)
      parameterSet reduce {
        case (l, r) =>
          ((l._1 + r._1), (l._2 + r._2), l._3 compose r._3, l._4 compose r._4)
      }
    }
    (lossAndParamSums._1, lossAndParamSums._2, lossAndParamSums._3, lossAndParamSums._4)
  }
}
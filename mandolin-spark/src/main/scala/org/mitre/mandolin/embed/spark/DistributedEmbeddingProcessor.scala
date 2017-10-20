package org.mitre.mandolin.embed.spark

import org.apache.spark.{SparkContext}
import org.apache.spark.Partitioner
import org.apache.spark.rdd.RDD
import org.mitre.mandolin.optimize.{TrainingUnitEvaluator, EpochProcessor}
import org.mitre.mandolin.optimize.spark.RandomPartitioner
import org.mitre.mandolin.embed._
import org.mitre.mandolin.util.Simple2DArray
import scala.reflect.ClassTag

/**
 * @author wellner
 */
class DistributedEmbeddingProcessor(sc: SparkContext,
    val evaluator: TrainingUnitEvaluator[SeqInstance, EmbedWeights, EmbedGradient, EmbedUpdater],    
    epochs: Int,
    workersPerPartition: Int,
    vocabSize: Int,
    embSize: Int,
    useAdaGrad: Boolean = false
  ) {
  
  def estimate(rdd: RDD[SeqInstance], mxEpochs: Option[Int] = None): EmbedWeights = {
    val numPartitions = rdd.partitions.length
    val t0 = System.nanoTime()
    var weights : Option[EmbedWeights] = None // initialWeights
    var updater : Option[EmbedUpdater] = None // initialUpdater
    // rdd.cache() // this is big, but usually fine to cache in memory
    // rdd.    
    val dataSize = rdd.count()
    for (i <- 0 until mxEpochs.getOrElse(epochs)) {
      val (newWeights, newUpdater) = processEpoch(i, rdd, numPartitions, weights, updater, vocabSize, embSize, dataSize)
      weights = Some(newWeights)
      updater = Some(newUpdater)
    }
    weights.get
  }

  def processEpoch(epochNum: Int, rdd: RDD[SeqInstance], numPartitions: Int, 
    currentWeights: Option[EmbedWeights], currentUpdater: Option[EmbedUpdater], vocabSize: Int, embSize: Int, dataSize: Long): (EmbedWeights, EmbedUpdater) = {
    // currentUpdater.compress()
    
    val compressedUpdater = currentUpdater map {u => u.compress()} 
    val bc = sc.broadcast((currentWeights match {case Some(w) => w case None => EmbedWeights(embSize, vocabSize)}, compressedUpdater))
    // val bcUpdater = sc.broadcast(currentUpdater)

    val _trainRDD = (rdd map {e => (e,1)})
    val trainRDD1 = _trainRDD.partitionBy(new RandomPartitioner(numPartitions))
    val trainRDD = trainRDD1.map(x => x._1)
    val arrayCutoff = vocabSize
    val _embSize = embSize
    val _useAdaGrad = useAdaGrad
    val _evaluator = evaluator
    val _wpp = workersPerPartition
    val _enum = epochNum
    val _ds = dataSize

    val parameterSet1 = trainRDD.mapPartitions({ insts =>
        println("*** About to process instances on worker node .. reading broadcasts")
        val (w,_u) = bc.value
        val u = _u map {u => u.decompress()}
        // val u = bcUpdater.value
        val ep = new EpochProcessor[SeqInstance, EmbedWeights, EmbedGradient, EmbedUpdater](_evaluator, workersPerPartition = _wpp, synchronous = false)// bcEp.value
        println("*** Finished reading broadcasts")
        val updater = u.getOrElse(if (_useAdaGrad) new EmbedAdaGradUpdater(0.1f, arrayCutoff, _embSize) else new NullUpdater(0.1f, 0.00001f))
        updater.resetLearningRates((_enum * _ds).toFloat)
        val partitionInsts = util.Random.shuffle(insts.toVector)
        w.accelerate
        val (loss, time) = ep.processPartitionWithinEpoch(1, partitionInsts, w, updater, 0L)   
        // val compressedUpdater = updater.compress()
        w.decelerate
        val wArrs = w.embW.ars.toSeq ++ w.outW.ars.toSeq
        val uArrs = updater.getArraySeq
        val allArrs = wArrs ++ uArrs
        val zipped = (0 until allArrs.length) zip allArrs
        zipped.toIterator
      }, true)
      
    // repartition to increase parallelism - may fix issues . . . 
    val parameterSet = parameterSet1.repartition(numPartitions * workersPerPartition)
    val aggregatedArrays =       
      parameterSet reduceByKey {
        case (l, r) => 
          var i = 0; while (i < l.length) {
            r(i) += l(i)
            i += 1
          }
          r
      }  
 
    val collectedArrays = aggregatedArrays.collect().sortWith{case ((a,_),(b,_)) => a < b}
    bc.destroy()
    // bcUpdater.destroy()
    println("RECEIVED Arrays: " + collectedArrays.length)
    val embArrays = new collection.mutable.ArrayBuffer[Array[Float]]
    val outArrays = new collection.mutable.ArrayBuffer[Array[Float]]
    val uEArrays = new collection.mutable.ArrayBuffer[Array[Float]]
    val uOArrays = new collection.mutable.ArrayBuffer[Array[Float]]
    var i = 0; while (i < collectedArrays.length) {
      val (arI,ar) = collectedArrays(i) 
      if (arI < arrayCutoff) embArrays append ar
      else if (arI < arrayCutoff*2) outArrays append ar
      else if (arI < arrayCutoff*3) uEArrays append ar
      else uOArrays append ar
      i += 1
    }
    val nEmb = new Simple2DArray(embArrays.toArray, vocabSize, embSize)
    val nOut = new Simple2DArray(outArrays.toArray, vocabSize, embSize)
    nEmb *= (1.0f / numPartitions)
    nOut *= (1.0f / numPartitions)
    
    val nEmbedding = new EmbedWeights(nEmb, nOut, 1.0f)
    val nUpdater : EmbedUpdater = if (useAdaGrad) {
      val nUEAr =  new Simple2DArray(uEArrays.toArray, vocabSize, embSize)
      val nUOAr =  new Simple2DArray(uOArrays.toArray, vocabSize, embSize)
      nUEAr *= (1.0f / numPartitions)
      nUOAr *= (1.0f / numPartitions)
      new EmbedAdaGradUpdater(0.1f, nUEAr, nUOAr) 
    } else new NullUpdater(0.1f / (1.0f + dataSize.toFloat * 0.00001f), 0.00001f)
    
    (nEmbedding, nUpdater)
  }
}
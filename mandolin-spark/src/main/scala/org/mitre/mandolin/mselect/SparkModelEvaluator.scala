package org.mitre.mandolin.mselect

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory
import org.mitre.mandolin.glp.{GLPComponentSet, GLPFactor}
import org.mitre.mandolin.glp.local.{LocalProcessor, LocalGLPOptimizer}
import org.mitre.mandolin.predict.local.{LocalEvalDecoder, LocalTrainer}
import org.mitre.mandolin.util.LocalIOAssistant
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

class SparkModelEvaluator(sc: SparkContext, trainBC: Broadcast[Vector[GLPFactor]], testBC: Broadcast[Vector[GLPFactor]]) 
extends ModelEvaluator with Serializable {
  override def evaluate(c: ModelConfig): Double = {
    val configRDD: RDD[ModelConfig] = sc.parallelize(Seq(c), 1)
    val _trainBC = trainBC
    val _testBC = testBC
    val accuracy = configRDD.mapPartitions { configIter =>
      val trData  = _trainBC.value
      val tstData = _testBC.value
      val accuracies = configIter map {config =>
        val learner = MandolinModelInstance(config)
        val acc = learner.train(trData, tstData)
        acc
      }
      accuracies
    }.collect()
    accuracy.toSeq(0)
  }
}

class SparkMxModelEvaluator(sc: SparkContext, trainBC: Broadcast[Vector[GLPFactor]], testBC: Broadcast[Vector[GLPFactor]]) 
extends ModelEvaluator with Serializable {

  override def evaluate(c: ModelConfig): Double = {
    val configRDD: RDD[ModelConfig] = sc.parallelize(Seq(c), 1)
    val _trainBC = trainBC
    val _testBC = testBC    
    val accuracy = configRDD.mapPartitions { configIter =>
      val trData  = _trainBC.value
      val tstData = _testBC.value
      val accuracies = configIter map {config =>
        val learner = MxModelInstance(config)
        val acc = learner.train(trData, tstData)
        acc
      }
      accuracies
    }.collect()
    accuracy.toSeq(0)
  }
}

class SparkMxFileSystemModelEvaluator(sc: SparkContext, trainData: String, testData: String) 
extends ModelEvaluator with Serializable {
  def evaluate (c: ModelConfig) : Double = {
    val configRDD: RDD[ModelConfig] = sc.parallelize(Seq(c), 1)
    val _trainData = trainData
    val _testData = testData
    val accuracy = configRDD.mapPartitions { configIter =>
      val accuracies = configIter map {config =>
        val learner = FileSystemImgMxModelInstance(config)
        val acc = learner.train(Vector(new java.io.File(_trainData)), Vector(new java.io.File(_testData)))
        acc        
      }
      accuracies
    }.collect()
    accuracy.toSeq(0)
  }
}

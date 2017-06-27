package org.mitre.mandolin.mselect

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory
import org.mitre.mandolin.mlp.{MMLPComponentSet, MMLPFactor}
import org.mitre.mandolin.mlp.standalone.{Processor, MMLPOptimizer}
import org.mitre.mandolin.predict.standalone.{EvalDecoder, Trainer}
import org.mitre.mandolin.util.LocalIOAssistant
import org.slf4j.LoggerFactory
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool
import scala.concurrent._
import java.util.concurrent.Executors

class SparkModelEvaluator(sc: SparkContext, trainBC: Broadcast[Vector[MMLPFactor]], testBC: Broadcast[Vector[MMLPFactor]])
  extends ModelEvaluator with Serializable {
  val logger = LoggerFactory.getLogger(this.getClass)

  override def evaluate(c: ModelConfig, generation: Int): (Double, Long) = {
    sc.setJobGroup("Generation " + generation, "["+c.toString()+"]", true)
    val configRDD: RDD[ModelConfig] = sc.parallelize(Seq(c), 1)
    val _trainBC = trainBC
    val _testBC = testBC
    val accuracy = configRDD.mapPartitions { configIter =>
      val trData  = _trainBC.value
      val tstData = _testBC.value
      val accuracies = configIter map {config =>
        val learner = MandolinModelInstance(config)

        val startTime = System.currentTimeMillis()
        val acc = learner.train(trData, Some(tstData))
        val endTime = System.currentTimeMillis()
        (acc, endTime - startTime)
      }
      accuracies
    }.collect()
  accuracy.toSeq(0)
  }

  def cancel(generation: Int) = {
    sc.cancelJobGroup("Generation " + generation)
  }
}

class SparkMxModelEvaluator(sc: SparkContext, trainBC: Broadcast[Vector[MMLPFactor]], testBC: Broadcast[Vector[MMLPFactor]])
  extends ModelEvaluator with Serializable {

  override def evaluate(c: ModelConfig, generation: Int): (Double, Long) = {
    val configRDD: RDD[ModelConfig] = sc.parallelize(Seq(c), 1)
    val _trainBC = trainBC
    val _testBC = testBC
    val accuracy = configRDD.mapPartitions { configIter =>
      val trData  = _trainBC.value
      val tstData = _testBC.value
      val accuracies = configIter map {config =>
        val learner = MxModelInstance(config)
        val startTime = System.currentTimeMillis()
        val acc = learner.train(trData, Some(tstData))
        val endTime = System.currentTimeMillis()
        (acc, endTime - startTime)
      }
      accuracies
    }.collect()
    accuracy.toSeq(0)
  }

  override def cancel(generation: Int): Unit = {}
}

class SparkMxFileSystemModelEvaluator(sc: SparkContext, trainData: String, testData: String)
  extends ModelEvaluator with Serializable {
  def evaluate(c: ModelConfig, generation: Int): (Double, Long) = {
    val configRDD: RDD[ModelConfig] = sc.parallelize(Seq(c), 1)
    val _trainData = trainData
    val _testData = testData
    val accuracy = configRDD.mapPartitions { configIter =>
      val accuracies = configIter map {config =>
        val learner = FileSystemImgMxModelInstance(config)
        val startTime = System.currentTimeMillis()
        val acc = learner.train(Vector(new java.io.File(_trainData)), Some(Vector(new java.io.File(_testData))))
        val endTime = System.currentTimeMillis()
        (acc, endTime - startTime)
      }
      accuracies
    }.collect()
    accuracy.toSeq(0)
  }
  def cancel(generation: Int) : Unit = {}
}

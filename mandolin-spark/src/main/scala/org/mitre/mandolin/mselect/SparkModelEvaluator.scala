package org.mitre.mandolin.mselect

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.mitre.mandolin.glp.{GLPComponentSet, GLPFactor}
import org.mitre.mandolin.glp.local.{LocalProcessor, LocalGLPOptimizer}
import org.mitre.mandolin.predict.local.{LocalEvalDecoder, LocalTrainer}
import org.mitre.mandolin.util.LocalIOAssistant
import org.slf4j.LoggerFactory
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

class SparkModelEvaluator(sc: SparkContext, trainBC: Broadcast[Vector[GLPFactor]], testBC: Broadcast[Vector[GLPFactor]]) extends ModelEvaluator with Serializable {
  val log = LoggerFactory.getLogger(getClass)

  override def evaluate(c: Seq[ModelConfig], generation: Int): Seq[Double] = {
    sc.setJobGroup("Generation " + generation, "["+c.map { config => "{"+config.toString()+"}" }.mkString(",")+"]", true)
    val startTime = System.currentTimeMillis()
    val configRDD: RDD[ModelConfig] = sc.parallelize(c, 1)
    val _trainBC = trainBC
    val _testBC = testBC
    val accuracy = configRDD.mapPartitions { configs =>
      val cv1 = configs.toList
      val cvec = cv1.par
      // set tasksupport to allocate N threads so each item is processed concurrently
      cvec.tasksupport_=(new ForkJoinTaskSupport(new ForkJoinPool(cvec.length)))
      val trData = _trainBC.value
      val tstData = _testBC.value
      val accuracies = cvec map { config =>
        val learner = GenericModelFactory.getLearnerInstance(config)
        val acc = learner.train(trData, tstData)
        acc
      }
      accuracies.toIterator
    }.collect()
    val stopTime = System.currentTimeMillis()
    val results = accuracy.toSeq
    results.foreach{ acc => log.info("accuracy:"+acc+" time:"+(stopTime-startTime))}
    results
  }

  def cancel(generation: Int) = {
    sc.cancelJobGroup("Generation " + generation)
  }
}


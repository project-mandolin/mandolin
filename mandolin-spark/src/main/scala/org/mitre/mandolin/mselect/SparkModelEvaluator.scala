package org.mitre.mandolin.mselect

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

class SparkModelEvaluator(sc: SparkContext, trainBC: Broadcast[Vector[String]], testBC: Broadcast[Vector[String]]) extends ModelEvaluator {
  override def evaluate(c: ModelConfig): Double = {
    val factory = new MandolinLogisticRegressionFactory
    val learner = factory.getLearnerInstance(c)
    val learnerRDD: RDD[LearnerInstance] = sc.parallelize(Array(learner))
    val accuracy = learnerRDD.map{ learner => learner.train(sc, trainBC, testBC) }.collect()
    accuracy(0)
  }

}


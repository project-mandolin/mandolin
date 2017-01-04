package org.mitre.mandolin.mselect

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.mitre.mandolin.glp.GLPComponentSet
import org.mitre.mandolin.glp.local.{LocalProcessor, LocalGLPOptimizer}
import org.mitre.mandolin.predict.local.{LocalEvalDecoder, LocalTrainer}
import org.mitre.mandolin.util.LocalIOAssistant

class SparkModelEvaluator(sc: SparkContext, trainBC: Broadcast[Vector[String]], testBC: Broadcast[Vector[String]]) extends ModelEvaluator with Serializable {
  override def evaluate(c: ModelConfig): Double = {
    val configRDD: RDD[ModelConfig] = sc.parallelize(Array(c), 1)
    val _trainBC = trainBC
    val _testBC = testBC
    val accuracy = configRDD.map { config => {
      val factory = new MandolinLogisticRegressionFactory
      val learner = factory.getLearnerInstance(config)
      val acc = learner.train(_trainBC, _testBC)
      acc
    }
    }.collect()
    accuracy(0)
  }

}


package org.mitre.mandolin.mselect

import org.mitre.mandolin.glp.{GLPComponentSet, GLPFactor}
import org.mitre.mandolin.glp.local.{LocalProcessor, LocalGLPOptimizer}
import org.mitre.mandolin.predict.local.{LocalEvalDecoder, LocalTrainer}
import org.mitre.mandolin.util.LocalIOAssistant
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

/**
 * ModelEvaluator is the class wrapping whatever functionality is required to train and test a
 * model configuration against a provided dataset, using x-validation, etc.
 */
abstract class ModelEvaluator {
  def evaluate(c: Seq[ModelConfig]) : Seq[(Double, Long)]
}

// XXX - this is for testing purposes only
class MockRandomModelEvaluator extends ModelEvaluator {
  
  private def pauseTime() = {
    val nsecs = util.Random.nextInt(10) * 1000
    Thread.sleep(nsecs)
  }
  
  def evaluate(c: Seq[ModelConfig]) : Seq[(Double, Long)] = {
    pauseTime() // simulate this taking a while
    c.map(_=>(util.Random.nextDouble(), util.Random.nextLong()))
  }
}

class LocalModelEvaluator(trData: Vector[GLPFactor], tstData: Vector[GLPFactor]) extends ModelEvaluator with Serializable {
  override def evaluate(c: Seq[ModelConfig]): Seq[(Double, Long)] = {

    val configs = c.toList
    val cvec = configs.par

    cvec.tasksupport_=(new ForkJoinTaskSupport(new ForkJoinPool(cvec.length)))
    val accuracies = cvec map {config =>
      val learner = MandolinModelInstance(config)
      val startTime = System.currentTimeMillis()
      val acc = learner.train(trData, tstData)
      val endTime = System.currentTimeMillis()
      (acc, endTime - startTime)
    }
    accuracies.seq
  }
}
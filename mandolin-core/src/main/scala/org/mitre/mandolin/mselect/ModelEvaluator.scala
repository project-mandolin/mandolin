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
  /**
    * Evaluate a sequence of configs, using a budget (usually number of iterations)
    */
  def evaluate(c: ModelConfig, generation:Int = 0): (Double, Long)
  def cancel(generation: Int)
}

// XXX - this is for testing purposes only
class MockRandomModelEvaluator extends ModelEvaluator {

  private def pauseTime() = {
    val nsecs = util.Random.nextInt(10) * 1000
    Thread.sleep(nsecs)
  }

  def evaluate(c: ModelConfig, generation : Int = 0): (Double, Long) = {
    pauseTime() // simulate this taking a while
    (util.Random.nextDouble(), util.Random.nextLong())
  }

  def cancel(generation: Int) {}
}

class LocalModelEvaluator(trData: Vector[GLPFactor], tstData: Option[Vector[GLPFactor]]) extends ModelEvaluator with Serializable {
  val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  
  override def evaluate(c: ModelConfig, generation: Int): (Double, Long) = {
    logger.info("Evaluating model configuration locally ...")
    val learner = MandolinModelInstance(c)
    logger.info("Acquired model evaluator instance ...")
    val startTime = System.currentTimeMillis()
    val acc = learner.train(trData, tstData)
    logger.info("Model evaluation complete (accuracy/result = " + acc + ")")
    val endTime = System.currentTimeMillis()
    (acc, endTime - startTime)
  }
   def cancel(generation: Int) {}
}
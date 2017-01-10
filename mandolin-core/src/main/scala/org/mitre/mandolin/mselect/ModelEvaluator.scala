package org.mitre.mandolin.mselect

/**
 * ModelEvaluator is the class wrapping whatever functionality is required to train and test a
 * model configuration against a provided dataset, using x-validation, etc.
 */
abstract class ModelEvaluator {
  def evaluate(c: Seq[ModelConfig]) : Seq[Double]
}

// XXX - this is for testing purposes only
class MockRandomModelEvaluator extends ModelEvaluator {
  
  private def pauseTime() = {
    val nsecs = util.Random.nextInt(10) * 1000
    Thread.sleep(nsecs)
  }
  
  def evaluate(c: Seq[ModelConfig]) : Seq[Double] = {
    pauseTime() // simulate this taking a while
    c.map(_=>util.Random.nextDouble())
  }
}
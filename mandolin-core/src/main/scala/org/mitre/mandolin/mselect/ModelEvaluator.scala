package org.mitre.mandolin.mselect

/**
 * ModelEvaluator is the class wrapping whatever functionality is required to train and test a
 * model configuration against a provided dataset, using x-validation, etc.
 */
abstract class ModelEvaluator {
  def evaluate(c: ModelConfig) : Double
}

class RandomModelEvaluator extends ModelEvaluator {
  def evaluate(c: ModelConfig) : Double = {
    util.Random.nextDouble()
  }
}
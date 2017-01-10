package org.mitre.mandolin.mselect

import scala.collection.mutable.ArrayBuffer
import WorkPullingPattern._

abstract class AcquisitionFunction {
  def score(config: ModelConfig) : Double
  def train(evalResults: ArrayBuffer[ModelEvalResult]) : Unit
}

abstract class ExpectedImprovement extends AcquisitionFunction {

}

class RandomAcquisitionFunction extends AcquisitionFunction {
  def score(config: ModelConfig) : Double = util.Random.nextDouble()
  def train(evalResults: ArrayBuffer[ModelEvalResult]) : Unit = {}
}

class MockAcquisitionFunction extends AcquisitionFunction {
  def score(config: ModelConfig) : Double = util.Random.nextDouble()
  def train(evalResults: ArrayBuffer[ModelEvalResult]) : Unit = {
    val ms = (util.Random.nextDouble() * 1 * 1000).toLong
    Thread.sleep(ms)
  }
}
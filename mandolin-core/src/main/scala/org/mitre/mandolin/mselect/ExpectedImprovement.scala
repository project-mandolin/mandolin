package org.mitre.mandolin.mselect

abstract class AcquisitionFunction {
  def score(config: ModelConfig) : Double
}

abstract class ExpectedImprovement extends AcquisitionFunction {

}

class RandomAcquisitionFunction extends AcquisitionFunction {
  def score(config: ModelConfig) : Double = util.Random.nextDouble()
}
package org.mitre.mandolin.gm

import org.mitre.mandolin.mlp._
import org.mitre.mandolin.optimize.{LossGradient, TrainingUnitEvaluator, Updater, Weights}
import org.mitre.mandolin.util.DenseTensor1

class MultiFactorWeights(val singleWeights: MMLPWeights, val pairWeights: MMLPWeights, m: Float) extends Weights[MultiFactorWeights](m) with Serializable {
  
  def compress(): Unit = {}
  def decompress(): Unit = {}
  def weightAt(i: Int) = throw new RuntimeException("Not implemented")
  
  val numWeights = singleWeights.numWeights + pairWeights.numWeights
  
  def compose(otherWeights: MultiFactorWeights) = {
    this *= mass
    otherWeights *= otherWeights.mass
    this ++ otherWeights
    val nmass = mass + otherWeights.mass
    this *= (1.0f / nmass)
    new MultiFactorWeights(this.singleWeights, this.pairWeights, nmass)
  }

  def add(otherWeights: MultiFactorWeights): MultiFactorWeights = {
    this += otherWeights
    this
  }

  def addEquals(otherWeights: MultiFactorWeights): Unit = {
    singleWeights addEquals otherWeights.singleWeights
    pairWeights addEquals otherWeights.pairWeights
  }

  def timesEquals(v: Float) = { 
    singleWeights.timesEquals(v)
    pairWeights.timesEquals(v)
  }

  def l2norm = throw new RuntimeException("Norm not implemented yet")

  def updateFromArray(ar: Array[Float]) = {
    if (singleWeights.wts.length > 1) throw new RuntimeException("array update for NN not implemented")
    else {
      val (m,b) = singleWeights.wts.get(0)      
      System.arraycopy(ar, 0, m.asArray, 0, m.getSize)
      System.arraycopy(ar, m.getSize, b.asArray, 0, b.getSize)
      val (mm, bb) = pairWeights.wts.get(0)
      System.arraycopy(ar, 0, mm.asArray, 0, mm.getSize)
      System.arraycopy(ar, mm.getSize, bb.asArray, 0, bb.getSize)
      
      this
    }
  }
  
  def updateFromArray(ar: Array[Double]) = {
    throw new RuntimeException("array update for NN not implemented")
  }

  def copy() = new MultiFactorWeights(singleWeights.copy(), pairWeights.copy(), m)

  def asArray() = throw new RuntimeException("Array form not available ")

  def asTensor1() = throw new RuntimeException("Tensor construction not available with deep weight array")
  
}

class MultiFactorLossGradient(l: Double, val singles: MMLPLayout, val pairs: MMLPLayout) extends LossGradient[MultiFactorLossGradient](l) {
  def add(other: MultiFactorLossGradient) = {
    singles.addEquals(other.singles)
    pairs.addEquals(other.pairs)
    new MultiFactorLossGradient(l + other.loss, singles, pairs)
  }
  
  def asArray = throw new RuntimeException("As Array not feasible with non-linear model")
}



class PairwiseFactorEvaluator[U <: Updater[MultiFactorWeights, MultiFactorLossGradient, U]](
    val singleGlp: ANNetwork, val pairGlp: ANNetwork, variableOrder: Int)
extends TrainingUnitEvaluator [MultiFactor, MultiFactorWeights, MultiFactorLossGradient, U] with Serializable {
  
  val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  
  val potentials = Array.tabulate(variableOrder){_ => Array.tabulate(variableOrder){_ => 0.0f}}
  val vec1 = Array.tabulate(variableOrder){_ => 0.0f}
  val vec2 = Array.tabulate(variableOrder){_ => 0.0f}
  val pairVec = Array.tabulate(variableOrder * variableOrder){ _ => 0.0f}
  
  var evaluated = 0
  
  def evaluateTrainingUnit(unit: MultiFactor, weights: MultiFactorWeights, u: U) : MultiFactorLossGradient = {
    
    val gradients = 
      unit.marginalInference(singleGlp, pairGlp, weights, potentials, vec1, vec2, pairVec, true)  // this does the heavy lift of full pair-wise inference and gradients

    /*
    val vstr = new StringBuilder
    pairVec foreach {v => vstr append v; vstr append ' '}
    logger.info("Pair vec: " + vstr.toString)
    
    val vstr1 = new StringBuilder
    vec1 foreach {v => vstr1 append v; vstr1 append ' '}
    logger.info("Vec1: " + vstr1.toString)
    
    val vstr2 = new StringBuilder
    vec2 foreach {v => vstr2 append v; vstr2 append ' '}
    logger.info("Vec2: " + vstr2.toString)
    */
    evaluated += 1
    if ((evaluated % 10000) == 0) logger.info("Processed [ " + evaluated + "] instances")       
    val (grPair, gr1) = gradients match {case Some((grPair, gr1)) => (grPair, gr1) case None => throw new RuntimeException("Expected gradients")}
    new MultiFactorLossGradient(0.0,gr1,grPair)
  }  
  
  def copy() = new PairwiseFactorEvaluator(singleGlp.sharedWeightCopy(), pairGlp.sharedWeightCopy(), variableOrder) // this copy will share weights but have separate layer data members
  
}

class MultiFactorAdaGradUpdater(val singles: MMLPAdaGradUpdater, val pairs: MMLPAdaGradUpdater, val initialLearningRate: Float = 0.9f, maxNormArray: Option[Array[Float]] = None,
    l1Array: Option[Array[Float]] = None, l2Array: Option[Array[Float]] = None)
  extends Updater[MultiFactorWeights, MultiFactorLossGradient, MultiFactorAdaGradUpdater] with Regularizer {

  singles.resetLearningRates(initialLearningRate)
  pairs.resetLearningRates(initialLearningRate)

  def asArray : Array[Float] = throw new RuntimeException("As array not available for complex updater")
  def updateFromArray(ar: Array[Float]) = throw new RuntimeException("From array not available for complex updater")
  def compress() = this
  def decompress() = this

  /*
   * A <i>shallow</i> copy so learning rates are shared across threads/partitions
   */
  def copy() = new MultiFactorAdaGradUpdater(singles.copy(), pairs.copy(), initialLearningRate, maxNormArray, l1Array, l2Array)

  def resetLearningRates(v: Float) = {
    singles.resetLearningRates(v)
    pairs.resetLearningRates(v)    
  }

  def compose(u: MultiFactorAdaGradUpdater) = {
    val c1 = singles.compose(u.singles)
    val c2 = pairs.compose(u.pairs)
    new MultiFactorAdaGradUpdater(c1,c2, initialLearningRate, maxNormArray, l1Array, l2Array)
  }

  def updateWeights(lossGrad: MultiFactorLossGradient, weights: MultiFactorWeights): Unit = {
    singles.updateWeights(new MMLPLossGradient(0.0,lossGrad.singles), weights.singleWeights)
    pairs.updateWeights(new MMLPLossGradient(0.0,lossGrad.pairs), weights.pairWeights)
  }
}

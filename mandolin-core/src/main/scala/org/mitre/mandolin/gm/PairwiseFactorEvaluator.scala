package org.mitre.mandolin.gm

import org.mitre.mandolin.glp.{GLPWeights, ANNetwork, GLPLossGradient, GLPFactor, GLPLayout, Regularizer, GLPAdaGradUpdater}
import org.mitre.mandolin.optimize.{Updater, TrainingUnitEvaluator, Weights, LossGradient}
import org.mitre.mandolin.util.DenseTensor1

class MultiFactorWeights(val singleWeights: GLPWeights, val pairWeights: GLPWeights, m: Float) extends Weights[MultiFactorWeights](m) with Serializable {
  
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

class MultiFactorLossGradient(l: Double, val singles: GLPLayout, val pairs: GLPLayout) extends LossGradient[MultiFactorLossGradient](l) {
  def add(other: MultiFactorLossGradient) = {
    singles.addEquals(other.singles)
    pairs.addEquals(other.pairs)
    new MultiFactorLossGradient(l + other.loss, singles, pairs)
  }
  
  def asArray = throw new RuntimeException("As Array not feasible with non-linear model")
}

class PairwiseFactorEvaluator[U <: Updater[MultiFactorWeights, MultiFactorLossGradient, U]](val singleGlp: ANNetwork, val pairGlp: ANNetwork)
extends TrainingUnitEvaluator [MultiFactor, MultiFactorWeights, MultiFactorLossGradient, U] with Serializable { 

  def evaluateTrainingUnit(unit: MultiFactor, weights: MultiFactorWeights, u: U) : MultiFactorLossGradient = {
    val pairUnit = unit.getInput
    val singleUnit1 = unit.singletons(0).getInput
    val singleUnit2 = unit.singletons(1).getInput
    singleGlp.forwardPass(singleUnit1.getInput, singleUnit1.getOutput, weights.singleWeights)
    val f1Output = singleGlp.outLayer.getOutput(true)
    singleGlp.forwardPass(singleUnit2.getInput, singleUnit2.getOutput, weights.singleWeights)
    val f2Output = singleGlp.outLayer.getOutput(true)
    pairGlp.forwardPass(pairUnit.getInput, pairUnit.getOutput, weights.pairWeights)
    val pairOutput = pairGlp.outLayer.getOutput(true)
    val a1 = f1Output.asArray
    val a2 = f2Output.asArray
    val dim = a1.length
    var sum = 0.0
    var maxScore = -Float.MaxValue
    val potentials = 
    Array.tabulate(dim){i =>
      Array.tabulate(dim){ j =>        
        val sc = a1(i) + a2(j) + unit.assignmentToIndex(Array(i,j))
        maxScore = math.max(maxScore, sc)
        sc
        }
      }    
    var i = 0; while (i < dim) {
      var j = 0; while (j < dim) {
        val potential = math.exp(potentials(i)(j) - maxScore)
        sum += potential
        potentials(i)(j) = potential.toFloat 
        j += 1
      }
      i += 1
    }
    i = 0; while (i < dim) {
      var j = 0; while (j < dim) {
        potentials(i)(j) /= sum.toFloat 
        j += 1
      }
      i += 1
    }
    val vec1 = Array.tabulate(dim){i =>
      var s = 0.0f
      val pi = potentials(i)
      var j = 0; while (j < dim) {
        s += pi(j)
        j += 1
      }
      s
    }
    // Re-run forward pass    
    singleGlp.forwardPass(singleUnit1.getInput, singleUnit1.getOutput, weights.singleWeights)
    singleGlp.outLayer.setOutput(new DenseTensor1(vec1)) // set this to the vec1
    val gr1 = singleGlp.backpropGradients(singleUnit1.getInput, singleUnit1.getOutput, weights.singleWeights)    
    val vec2 = Array.tabulate(dim){j =>
      var s = 0.0f
      val pj = potentials(j)
      var i = 0; while (i < dim) {
        s += pj(i)
        i += 1
      }
      s
    }
    singleGlp.forwardPass(singleUnit2.getInput, singleUnit2.getOutput, weights.singleWeights)
    singleGlp.outLayer.setOutput(new DenseTensor1(vec2)) // set this to the vec2
    val gr2 = singleGlp.backpropGradients(singleUnit2.getInput, singleUnit2.getOutput, weights.singleWeights)
    
    pairGlp.forwardPass(pairUnit.getInput, pairUnit.getOutput, weights.pairWeights)
    
    // flatten the potentials back to a single vector
    val pairVec = Array.tabulate(unit.numConfigs){cInd =>
      val assignment = unit.indexToAssignment(cInd)
      potentials(assignment(0))(assignment(1))
    }
    
    pairGlp.outLayer.setOutput(new DenseTensor1(pairVec))
    val grPair = pairGlp.backpropGradients(pairUnit.getInput, pairUnit.getOutput, weights.pairWeights)
    gr1.addEquals(gr2, 1.0f)
    new MultiFactorLossGradient(0.0,gr1,grPair)
  }  
  
  def copy() = new PairwiseFactorEvaluator(singleGlp.sharedWeightCopy(), pairGlp.sharedWeightCopy()) // this copy will share weights but have separate layer data members
  
}

class MultiFactorAdaGradUpdater(val singles: GLPAdaGradUpdater, val pairs: GLPAdaGradUpdater, val initialLearningRate: Float = 0.9f, maxNormArray: Option[Array[Float]] = None,
    l1Array: Option[Array[Float]] = None, l2Array: Option[Array[Float]] = None)
  extends Updater[MultiFactorWeights, MultiFactorLossGradient, MultiFactorAdaGradUpdater] with Regularizer {

  singles.resetLearningRates(initialLearningRate)
  pairs.resetLearningRates(initialLearningRate)
  // sumSquaredSingles set initialLearningRate // set sum squared to initial learning rate
  // sumSquaredPairs set initialLearningRate

  // val nLayers = sumSquaredSingles.length 
  
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
    

  @inline
  final private def fastSqrt(x: Double) =
    java.lang.Double.longBitsToDouble(((java.lang.Double.doubleToLongBits(x) >> 32) + 1072632448) << 31)

    
  @inline
  final private def fastSqrt(x: Float) : Float = 
    java.lang.Float.intBitsToFloat(532483686 + (java.lang.Float.floatToRawIntBits(x) >> 1))
      
  def updateWeights(lossGrad: MultiFactorLossGradient, weights: MultiFactorWeights): Unit = {
    singles.updateWeights(new GLPLossGradient(0.0,lossGrad.singles), weights.singleWeights)
    pairs.updateWeights(new GLPLossGradient(0.0,lossGrad.pairs), weights.pairWeights)
  }
}

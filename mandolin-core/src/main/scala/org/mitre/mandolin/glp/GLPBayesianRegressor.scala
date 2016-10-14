package org.mitre.mandolin.glp

import org.mitre.mandolin.predict.{ EvalPredictor, RegressionConfusion }
import breeze.linalg.{ DenseMatrix => BreezeMat, DenseVector => BreezeVec }
import breeze.linalg.{inv, diag}
import breeze.numerics._

class GLPBayesianRegressor(network: ANNetwork, 
    designMatrix: BreezeMat[Double], 
    designTargets: BreezeVec[Double], 
    beta: Double,  // needs to be integrated out in subsequent version
    alpha: Double, // ditto 
    getScores: Boolean = false)
  extends EvalPredictor[GLPFactor, GLPWeights, (Double, Double), RegressionConfusion] with Serializable { 
  
  private val size = designMatrix.rows
  
  val meanFn = 0.0
  
  val meanSubtractedTargets = designTargets - meanFn
    
  
  val designInv = inv(designMatrix)
  val designTrans = designMatrix.t
    
    
  val K = beta * (designTrans * designMatrix) + diag(BreezeVec.fill(size){alpha})
  val kInv = inv(K)
  val m =  (kInv * designMatrix.t * meanSubtractedTargets) * beta  

  def getPrediction(u: GLPFactor, wts: GLPWeights): (Double, Double) = {
    network.forwardPass(u.getInput, u.getOutput, wts, false)
    val numLayers = network.layers.size  
    val penultimateOutput = network.layers(numLayers - 2).getOutput(false).asArray
    val basis = BreezeVec.tabulate[Double](penultimateOutput.length){(i: Int) => penultimateOutput(i).toDouble}
    val predMean = (m.t * basis) + meanFn
    val predVar  = basis.t * kInv * basis + (1.0 / beta)
    (predMean, predVar)
  }

  def getScoredPredictions(u: GLPFactor, w: GLPWeights): Seq[(Float, (Double, Double))] = {
    throw new RuntimeException("Score predictions don't make sense with continuous outputs")
  }

  def getLoss(u: GLPFactor, w: GLPWeights): Double = {
    // network.forwardPass(u.getInput, u.getOutput, w, false)
    // network.getCost
    0.0
  }

  def getConfusion(u: GLPFactor, w: GLPWeights): RegressionConfusion = {
    throw new RuntimeException("Confusion doesn't make sense for continuous outputs")
  }
}
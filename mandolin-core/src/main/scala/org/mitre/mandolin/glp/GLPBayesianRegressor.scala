package org.mitre.mandolin.glp

import org.mitre.mandolin.predict.{ EvalPredictor, RegressionConfusion }
import breeze.linalg.{ DenseMatrix => BreezeMat, DenseVector => BreezeVec }
import breeze.linalg.{inv, diag}
import breeze.numerics._
import org.mitre.mandolin.glp.local.LocalGLPModelReader
import org.mitre.mandolin.util.LocalIOAssistant

class GLPBayesianRegressor(network: ANNetwork, 
    designMatrix: BreezeMat[Double], 
    designTargets: BreezeVec[Double], 
    beta: Double,  // needs to be integrated out in subsequent version
    alpha: Double, // ditto 
    getScores: Boolean = false)
  extends EvalPredictor[GLPFactor, GLPWeights, (Double, Double), RegressionConfusion] with Serializable { 
  
  private val size = designMatrix.rows
  
  val meanFn = 0.0
  
  val meanSubtractedTargets = designTargets - meanFn  // y tilde from paper
      
  val designTrans = designMatrix.t
    
  val inDim = designMatrix.cols
  val K = beta * (designTrans * designMatrix) + diag(BreezeVec.fill(inDim){alpha})
  val kInv = inv(K)
  val m =  (kInv * designMatrix.t * meanSubtractedTargets) * beta  

  def getPrediction(u: GLPFactor, wts: GLPWeights): (Double, Double) = {
    val numLayers = network.layers.size
    if (numLayers > 1) network.forwardPass(u.getInput, u.getOutput, wts, false)      
    val penultimateOutput = if (numLayers > 1) network.layers(numLayers - 2).getOutput(false).asArray else u.getInput.asArray
    val basis = BreezeVec.tabulate[Double](penultimateOutput.length){(i: Int) => penultimateOutput(i).toDouble} // convert array to Breeze vector
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

/**
 * Main object to run direct tests/examples using Bayesian regression posterior inference
 */
object GLPBayesianRegressor {
  
  def main(args: Array[String]) : Unit = {
    val fp = args(0)
    val data = new java.io.File(args(1))
    val reader = new LocalGLPModelReader
    val io = new LocalIOAssistant
    val model = reader.readModel(fp, io)
    val fe = model.fe
    val dataFactors = scala.io.Source.fromFile(data).getLines.toList map { l =>
      fe.extractFeatures(l)
    }
    val dfInVecs = dataFactors map {x =>
      val inV = x.getInput    
      val v = BreezeVec.tabulate(inV.getDim){i => inV(i).toDouble}
      BreezeMat(v)
      }
    val bMat = dfInVecs.reduce{(a,b) => BreezeMat.vertcat(a,b)}
    val dfArray = dataFactors.toArray
    val targetsVec = BreezeVec.tabulate(dataFactors.length){i => dfArray(i).getOutput(0).toDouble}
    val predictor = new GLPBayesianRegressor(model.ann, bMat, targetsVec, 1.0, 1.0, false)
    val freqPredictor = new RegressionGLPPredictor(model.ann, true)
    val oc = new GLPRegressionOutputConstructor    
    val evalDecoder = new org.mitre.mandolin.predict.local.LocalDecoder(fe, predictor, oc)
    dataFactors foreach {l =>
      val (prMean, prVar) = evalDecoder.pr.getPrediction(l, model.wts)
      val freqMean = freqPredictor.getPrediction(l, model.wts)
      println("Predicted mean: " + prMean + " Var: " + prVar + "   ==> Actual value: " + l.getOutput(0) + " ++ Freq mean: " + freqMean(0).toString)
      }
  }
}
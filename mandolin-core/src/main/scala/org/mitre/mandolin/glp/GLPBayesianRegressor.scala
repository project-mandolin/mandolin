package org.mitre.mandolin.glp

import org.mitre.mandolin.predict.{ OutputConstructor, EvalPredictor, RegressionConfusion }
import breeze.linalg.{ DenseMatrix => BreezeMat, DenseVector => BreezeVec }
import breeze.linalg.{pinv, diag, sum, inv}
import breeze.numerics._
import org.mitre.mandolin.glp.local.LocalGLPModelReader
import org.mitre.mandolin.util.LocalIOAssistant

class GLPBayesianRegressor(network: ANNetwork, 
    designMatrix: BreezeMat[Double], 
    designTargets: BreezeVec[Double], 
    varX: Double,  // needs to be integrated out in subsequent version
    alpha: Double, // ditto 
    getScores: Boolean = false)
  extends EvalPredictor[GLPFactor, GLPWeights, (Double, Double), RegressionConfusion] with Serializable { 
  
  private val size = designMatrix.rows
  private val numFeatures = designMatrix.cols
  
  val meanFn = 0.0
  
  val meanSubtractedTargets = designTargets - meanFn  // y tilde from paper
      
  val designTrans = designMatrix.t
  
  println("DesignMatrix dims = " + designMatrix)

  val freqW = 
    if (true) {
      pinv(designMatrix) * designTargets
    } else {
      val rr = designTrans * designMatrix
      println("Inverting rr = ")
      println(rr.toString)
      inv(rr) * designMatrix.t * designTargets // least squares estimate of weights
    }  
  
  val yAvg = sum(designTargets) / size
  val xAvg = (designTrans * BreezeVec.ones[Double](size)) / size.toDouble
  // val freqB = yAvg - (freqW.t * xAvg) // bias/intercept
  
  val diffs = (designMatrix * freqW) - designTargets
  val variance = if (varX > 0.0) varX else sum(diffs :* diffs) / size // this is inverse variance
  val varInv = 1.0 / variance
    
  val inDim = designMatrix.cols
  val K = varInv * (designTrans * designMatrix) + diag(BreezeVec.fill(inDim){alpha})
  println("Inverting K =") 
  println(K.toString())
  val kInv = inv(K)  
  val m =  (kInv * designMatrix.t * meanSubtractedTargets) * varInv
  //val m = (kInv * meanSubtractedTargets) * varInv
  
  // println("FreqW => " + freqW.toString)


  def getPrediction(u: GLPFactor, wts: GLPWeights): (Double, Double) = {
    val numLayers = network.layers.size
    if (numLayers > 1) network.forwardPass(u.getInput, u.getOutput, wts, false)      
    val penultimateOutput = if (numLayers > 1) network.layers(numLayers - 2).getOutput(false).asArray else u.getInput.asArray
    // add the bias input as first element of basis vector
    val basis = BreezeVec.tabulate[Double](penultimateOutput.length + 1){(i: Int) => if (i > 0) penultimateOutput(i-1).toDouble else 1.0}
    // println("Basis = " + basis)
    val predMean = (m.t * basis) + meanFn
    val predVar  = basis.t * kInv * basis + variance
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
    val fpModel = args(0)
    val data = new java.io.File(args(1))
    val reader = new LocalGLPModelReader
    val io = new LocalIOAssistant
    val model = reader.readModel(fpModel, io)
    val fe = model.fe
    val dataFactors = scala.io.Source.fromFile(data).getLines.toList map { l =>
      fe.extractFeatures(l)
    }
    val glp = model.ann
    val dfInVecs = dataFactors map {x =>
      val inV = if (glp.layers.length == 1) x.getInput else {
        glp.forwardPass(x.getInput, x.getOutput, model.wts, false)
        glp.layers(glp.layers.length - 2).getOutput(false)
      }    
      val v = BreezeVec.tabulate(inV.getDim + 1){i => if (i > 0) inV(i-1).toDouble else 1.0} // add in bias to designmatrix
      BreezeMat(v)
      }
    val bMat = dfInVecs.reduce{(a,b) => BreezeMat.vertcat(a,b)}  // the design matrix
    val dfArray = dataFactors.toArray
    val targetsVec = BreezeVec.tabulate(dataFactors.length){i => dfArray(i).getOutput(0).toDouble} // the target vector
    println("Targets vec = " + targetsVec)
    val predictor = new GLPBayesianRegressor(model.ann, bMat, targetsVec, 0.0, 0.0, false)    
    dataFactors foreach {l =>
      val (prMean, prVar) = predictor.getPrediction(l, model.wts)
      // val freqMean = freqPredictor.getPrediction(l, model.wts)
      println("Predicted mean: " + prMean + " Var: " + prVar + "   ==> Actual value: " + l.getOutput(0))
      }
    //val oc = new GLPRegressionOutputConstructor    
    //val evalDecoder = new org.mitre.mandolin.predict.local.LocalDecoder(fe, predictor, oc)    
    // val freqPredictor = new RegressionGLPPredictor(model.ann, true)
  }  
}
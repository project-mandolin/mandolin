package org.mitre.mandolin.mlp

import org.mitre.mandolin.predict.{EvalPredictor, RegressionConfusion}
import breeze.linalg.{DenseMatrix => BreezeMat, DenseVector => BreezeVec}
import breeze.linalg.{pinv, diag, sum, inv}
import org.mitre.mandolin.mlp.standalone.StandaloneMMLPModelReader
import org.mitre.mandolin.util.LocalIOAssistant

class MMLPBayesianRegressor(network: ANNetwork,
                            designMatrix: BreezeMat[Double],
                            designTargets: BreezeVec[Double],
                            varX: Double, // needs to be integrated out in subsequent version
                            alpha: Double, // ditto
                            getScores: Boolean = false)
  extends EvalPredictor[MMLPFactor, MMLPWeights, (Double, Double), RegressionConfusion] with Serializable {

  private val size = designMatrix.rows
  private val numFeatures = designMatrix.cols

  val meanFn = 0.0

  val meanSubtractedTargets = designTargets - meanFn // y tilde from paper

  val designTrans = designMatrix.t

  val freqW =
    if (true) {
      pinv(designMatrix) * designTargets
    } else {
      val rr = designTrans * designMatrix
      inv(rr) * designMatrix.t * designTargets // least squares estimate of weights
    }

  val yAvg = sum(designTargets) / size
  val xAvg = (designTrans * BreezeVec.ones[Double](size)) / size.toDouble
  // val freqB = yAvg - (freqW.t * xAvg) // bias/intercept

  val diffs = (designMatrix * freqW) - designTargets
  val variance = if (varX > 0.0) varX else sum(diffs :* diffs) / size // this is inverse variance
  val varInv = 1.0 / variance

  // make this a public method as this part may be re-used by techniques that look at updated variance..
  def getKInv(designMat: BreezeMat[Double]) = {
    val inDim = designMat.cols
    val designMatTrans = designMat.t
    val K = varInv * (designMatTrans * designMat) + diag(BreezeVec.fill(inDim) {
      alpha
    })
    inv(K)
  }

  val kInv = getKInv(designMatrix)

  val m = (kInv * designMatrix.t * meanSubtractedTargets) * varInv

  def getUpdatedVariance(basis: BreezeVec[Double], designMat: BreezeMat[Double]): Double = {
    val kinv = getKInv(designMat)
    basis.t * kinv * basis + variance
  }

  def getBasisVector(u: MMLPFactor, wts: MMLPWeights): BreezeVec[Double] = {
    val numLayers = network.layers.size
    if (numLayers > 1) network.forwardPass(u.getInput, u.getOutput, wts, false)
    val penultimateOutput = if (numLayers > 1) network.layers(numLayers - 2).getOutput(false).asArray else u.getInput.asArray
    // add the bias input as first element of basis vector
    // add a bit of noise to avoid singular matrices
    val basis = BreezeVec.tabulate[Double](penultimateOutput.length + 1) { (i: Int) =>
      if (i > 0) {
        val v = penultimateOutput(i - 1).toDouble
        if (v > 0.0) v - (math.random * .04) else v + (math.random * 0.04)
      } else 1.0
    }
    basis
  }

  def getPrediction(basis: BreezeVec[Double], wts: MMLPWeights): (Double, Double) = {
    val predMean = (m.t * basis) + meanFn
    val predVar = basis.t * kInv * basis + variance
    (predMean, predVar)
  }

  def getPrediction(u: MMLPFactor, wts: MMLPWeights): (Double, Double) = {
    val basis = getBasisVector(u, wts)
    getPrediction(basis, wts)
  }

  def getScoredPredictions(u: MMLPFactor, w: MMLPWeights): Seq[(Float, (Double, Double))] = {
    throw new RuntimeException("Score predictions don't make sense with continuous outputs")
  }

  def getLoss(u: MMLPFactor, w: MMLPWeights): Double = {
    // network.forwardPass(u.getInput, u.getOutput, w, false)
    // network.getCost
    0.0
  }

  def getConfusion(u: MMLPFactor, w: MMLPWeights): RegressionConfusion = {
    throw new RuntimeException("Confusion doesn't make sense for continuous outputs")
  }
}


/**
  * Main object to run direct tests/examples using Bayesian regression posterior inference
  */
object MMLPBayesianRegressor {

  def main(args: Array[String]): Unit = {
    val fpModel = args(0)
    val data = new java.io.File(args(1))
    val reader = new StandaloneMMLPModelReader
    val io = new LocalIOAssistant
    val model = reader.readModel(fpModel, io)
    val fe = model.fe
    val dataFactors = scala.io.Source.fromFile(data).getLines.toList map { l =>
      fe.extractFeatures(l)
    }
    val glp = model.ann
    val dfInVecs = dataFactors map { x =>
      val inV = if (glp.layers.length == 1) x.getInput else {
        glp.forwardPass(x.getInput, x.getOutput, model.wts, false)
        glp.layers(glp.layers.length - 2).getOutput(false)
      }
      val v = BreezeVec.tabulate(inV.getDim + 1) { i => if (i > 0) inV(i - 1).toDouble else 1.0 } // add in bias to designmatrix
      BreezeMat(v)
    }
    val bMat = dfInVecs.reduce { (a, b) => BreezeMat.vertcat(a, b) } // the design matrix
    val dfArray = dataFactors.toArray
    val targetsVec = BreezeVec.tabulate(dataFactors.length) { i => dfArray(i).getOutput(0).toDouble } // the target vector
    println("Targets vec = " + targetsVec)
    val predictor = new MMLPBayesianRegressor(model.ann, bMat, targetsVec, 0.0, 0.0, false)
    dataFactors foreach { l =>
      val (prMean, prVar) = predictor.getPrediction(l, model.wts)
      println("Predicted mean: " + prMean + " Var: " + prVar + "   ==> Actual value: " + l.getOutput(0))
    }
  }
}
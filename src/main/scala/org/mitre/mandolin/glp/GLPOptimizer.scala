package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.optimize.local.{LocalOnlineOptimizer, LocalBatchOptimizer}
import org.mitre.mandolin.optimize.{OnlineOptimizer, Params, BatchEvaluator, GenData, RDDData, VectorData}
import org.apache.spark.SparkContext

class GlpLocalBatchEvaluator(ev: GLPInstanceEvaluator) extends BatchEvaluator[GLPFactor, GLPWeights, GLPLossGradient] {
  def evaluate(gd:GenData[GLPFactor], w: GLPWeights) : GLPLossGradient = {
    gd match {
      case data: VectorData[GLPFactor] =>
        data.vec map {d => ev.evaluateTrainingUnit(d, w)} reduce {_ ++ _}
      case _ : RDDData[GLPFactor] => throw new RuntimeException("Require local Scala vector data sequence for LocalBatchEvaluator")
    }
  }
}

/**
 * @author wellner
 */
object GLPOptimizer {
  
  def getDistributedOptimizer(sc: SparkContext, appSettings: GLPModelSettings, network: ANNetwork, evaluator: GLPInstanceEvaluator) = {
    val weights = network.generateRandomWeights
    val composeStrategy = appSettings.composeStrategy match {
      case "maximum" => Maximum
      case "average" => Average
      case _ => Minimum
    }
    val l1ArrayVals = network.layers map {l => l.ltype.l1}
    val l2ArrayVals = network.layers map {l => l.ltype.l2}
    val mnArrayVals = network.layers map {l => l.ltype.maxNorm}
    val l1Array = if (l1ArrayVals.max > 0.0) Some(l1ArrayVals.toArray) else None
    val l2Array = if (l2ArrayVals.max > 0.0) Some(l2ArrayVals.toArray) else None
    val maxNormArray = if (mnArrayVals.max > 0.0) Some(mnArrayVals.toArray) else None
    
    appSettings.method match {
      case "adagrad" =>
        val sumSquared = network.generateZeroedLayout
        sumSquared set appSettings.initialLearnRate // set to the initial learning rate
        val updater = new GLPAdaGradUpdater(sumSquared, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array=l1Array, l2Array=l2Array,
            compose = composeStrategy)
        new OnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPAdaGradUpdater](sc, weights, evaluator, updater, appSettings)
      case "adadelta" =>
        val sumSquared = network.generateZeroedLayout
        val prevUpdates = network.generateZeroedLayout
        val up = new GLPAdaDeltaUpdater(sumSquared, prevUpdates, appSettings.epsilon, appSettings.rho, compose = composeStrategy, maxNorm=appSettings.maxNorm)
        new OnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPAdaDeltaUpdater](sc, weights, evaluator, up, appSettings)
      case "rmsprop" =>
        val sumSquared = network.generateZeroedLayout
        sumSquared set appSettings.initialLearnRate // set to the initial learning rate
        val updater = new GLPRMSPropUpdater(sumSquared, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array=l1Array, l2Array=l2Array,
            compose = composeStrategy)
        new OnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPRMSPropUpdater](sc, weights, evaluator, updater, appSettings)
      case "nasgd" => // Nesterov accelerated
        val momentum = network.generateZeroedLayout
        val uu = new GLPSgdUpdater(momentum, true, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array=l1Array, l2Array=l2Array,
            compose = composeStrategy)
        new OnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPSgdUpdater](sc, weights, evaluator, uu, appSettings)      
      case "adam" =>
        val mom1 = network.generateZeroedLayout
        val mom2 = network.generateZeroedLayout
        val uu = new GLPAdamUpdater(0.001, 0.9, 0.999, mom1, mom2, maxNormArray=maxNormArray, l1Array = l1Array, l2Array = l2Array,
            composeSt = composeStrategy)
        new OnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPAdamUpdater](sc, weights, evaluator, uu, appSettings)
      case "sgd" =>
        val up = new BasicGLPSgdUpdater(appSettings.initialLearnRate)
        new OnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, BasicGLPSgdUpdater](sc, weights, evaluator, up, appSettings)
      case a => throw new RuntimeException("Unrecognized onlien training method: " + a)
    }
  }
  
  def getLocalOptimizer(appSettings: GLPModelSettings, evaluator: GLPInstanceEvaluator) = {
    val network = evaluator.glp
    val weights = network.generateRandomWeights
    val sumSquared = network.generateZeroedLayout
    val composeStrategy = appSettings.composeStrategy match {
      case "maximum" => Maximum
      case "average" => Average
      case _ => Minimum
    }
    val l1ArrayVals = network.layers map {l => l.ltype.l1}
    val l2ArrayVals = network.layers map {l => l.ltype.l2}
    val mnArrayVals = network.layers map {l => l.ltype.maxNorm}
    val l1Array = if (l1ArrayVals.max > 0.0) Some(l1ArrayVals.toArray) else None
    val l2Array = if (l2ArrayVals.max > 0.0) Some(l2ArrayVals.toArray) else None
    val maxNormArray = if (mnArrayVals.max > 0.0) Some(mnArrayVals.toArray) else None
    
    appSettings.method match {
      case "adagrad" =>
        sumSquared set appSettings.initialLearnRate // set to the initial learning rate
        val updater = new GLPAdaGradUpdater(sumSquared, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array = l1Array, l2Array = l2Array, 
            compose = composeStrategy)
        new LocalOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPAdaGradUpdater](weights, evaluator, updater, appSettings)
      case "adadelta" =>
        val prevUpdates = network.generateZeroedLayout
        val uu = new GLPAdaDeltaUpdater(sumSquared, prevUpdates, appSettings.epsilon, appSettings.rho,  
            compose = composeStrategy, maxNorm=appSettings.maxNorm)
        new LocalOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPAdaDeltaUpdater](weights, evaluator, uu, appSettings)
      case "rmsprop" =>
        val sumSquared = network.generateZeroedLayout
        sumSquared set appSettings.initialLearnRate // set to the initial learning rate
        val updater = new GLPRMSPropUpdater(sumSquared, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array = l1Array, l2Array = l2Array,
            compose = composeStrategy)
        new LocalOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPRMSPropUpdater](weights, evaluator, updater, appSettings)
      case "nasgd" => // Nesterov accelerated momentum-based SGD
        val momentum = network.generateZeroedLayout
        val uu = new GLPSgdUpdater(momentum, true, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array = l1Array, l2Array = l2Array,
            compose = composeStrategy)
        new LocalOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPSgdUpdater](weights, evaluator, uu, appSettings) 
      case "sgd" =>
        val uu = new BasicGLPSgdUpdater(appSettings.initialLearnRate)
        new LocalOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, BasicGLPSgdUpdater](weights, evaluator, uu, appSettings)
      case "adam" =>
        val mom1 = network.generateZeroedLayout
        val mom2 = network.generateZeroedLayout
        val uu = new GLPAdamUpdater(0.001, 0.9, 0.999, mom1, mom2, maxNormArray=maxNormArray, l1Array = l1Array, l2Array = l2Array,
            composeSt = composeStrategy)
        new LocalOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPAdamUpdater](weights, evaluator, uu, appSettings)
      case "lbfgs" =>
        val localBatchEvaluator = new GlpLocalBatchEvaluator(evaluator)
        val dim = weights.numWeights
        new LocalBatchOptimizer[GLPFactor, GLPWeights, GLPLossGradient](dim, weights, localBatchEvaluator, new Params())
      case a => throw new RuntimeException("Unrecognized online training method: " + a)
    }
  }

}


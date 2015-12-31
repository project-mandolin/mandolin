package org.mitre.mandolin.glp.spark

import org.mitre.mandolin.glp._
import org.mitre.mandolin.optimize.spark.DistributedOnlineOptimizer
import org.apache.spark.SparkContext

object DistributedGLPOptimizer {

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
        new DistributedOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPAdaGradUpdater](sc, weights, evaluator, updater, appSettings)
      case "adadelta" =>
        val sumSquared = network.generateZeroedLayout
        val prevUpdates = network.generateZeroedLayout
        val up = new GLPAdaDeltaUpdater(sumSquared, prevUpdates, appSettings.epsilon, appSettings.rho, compose = composeStrategy, maxNorm=appSettings.maxNorm)
        new DistributedOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPAdaDeltaUpdater](sc, weights, evaluator, up, appSettings)
      case "rmsprop" =>
        val sumSquared = network.generateZeroedLayout
        sumSquared set appSettings.initialLearnRate // set to the initial learning rate
        val updater = new GLPRMSPropUpdater(sumSquared, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array=l1Array, l2Array=l2Array,
            compose = composeStrategy)
        new DistributedOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPRMSPropUpdater](sc, weights, evaluator, updater, appSettings)
      case "nasgd" => // Nesterov accelerated
        val momentum = network.generateZeroedLayout
        val uu = new GLPSgdUpdater(momentum, true, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array=l1Array, l2Array=l2Array,
            compose = composeStrategy)
        new DistributedOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPSgdUpdater](sc, weights, evaluator, uu, appSettings)      
      case "adam" =>
        val mom1 = network.generateZeroedLayout
        val mom2 = network.generateZeroedLayout
        val uu = new GLPAdamUpdater(0.001f, 0.9f, 0.999f, mom1, mom2, maxNormArray=maxNormArray, l1Array = l1Array, l2Array = l2Array,
            composeSt = composeStrategy)
        new DistributedOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPAdamUpdater](sc, weights, evaluator, uu, appSettings)
      case "sgd" =>
        val up = new BasicGLPSgdUpdater(appSettings.initialLearnRate)
        new DistributedOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, BasicGLPSgdUpdater](sc, weights, evaluator, up, appSettings)
      case a => throw new RuntimeException("Unrecognized onlien training method: " + a)
    }
  }

}
package org.mitre.mandolin.mlp.standalone

import org.mitre.mandolin.mlp._
import org.mitre.mandolin.optimize.standalone.OnlineOptimizer

object MMLPOptimizer {
  
  def getOptimizer(network: ANNetwork, method: String, lr: Float, numEpochs: Int) = {
    
  }
  
  def getOptimizer(appSettings: MandolinMLPSettings, network: ANNetwork) = {
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
        val updater = new MMLPAdaGradUpdater(sumSquared, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array = l1Array, l2Array = l2Array,
            compose = composeStrategy)
        val evaluator = new MMLPInstanceEvaluator[MMLPAdaGradUpdater](network)
        new OnlineOptimizer[MMLPFactor, MMLPWeights, MMLPLossGradient, MMLPAdaGradUpdater](weights, evaluator, updater, appSettings)
      case "adadelta" =>
        val prevUpdates = network.generateZeroedLayout
        val uu = new MMLPAdaDeltaUpdater(sumSquared, prevUpdates, appSettings.epsilon, appSettings.rho,
            compose = composeStrategy, maxNorm=appSettings.maxNorm)
        val evaluator = new MMLPInstanceEvaluator[MMLPAdaDeltaUpdater](network)
        new OnlineOptimizer[MMLPFactor, MMLPWeights, MMLPLossGradient, MMLPAdaDeltaUpdater](weights, evaluator, uu, appSettings)
      case "rmsprop" =>
        val sumSquared = network.generateZeroedLayout
        sumSquared set appSettings.initialLearnRate // set to the initial learning rate
        val updater = new MMLPRMSPropUpdater(sumSquared, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array = l1Array, l2Array = l2Array,
            compose = composeStrategy)
        val evaluator = new MMLPInstanceEvaluator[MMLPRMSPropUpdater](network)
        new OnlineOptimizer[MMLPFactor, MMLPWeights, MMLPLossGradient, MMLPRMSPropUpdater](weights, evaluator, updater, appSettings)
      case "nasgd" => // Nesterov accelerated momentum-based SGD
        val momentum = network.generateZeroedLayout
        val uu = new MMLPSgdUpdater(momentum, true, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array = l1Array, l2Array = l2Array,
            compose = composeStrategy)
        val evaluator = new MMLPInstanceEvaluator[MMLPSgdUpdater](network)
        new OnlineOptimizer[MMLPFactor, MMLPWeights, MMLPLossGradient, MMLPSgdUpdater](weights, evaluator, uu, appSettings)
      case "sgd" =>
        val uu = new BasicMMLPSgdUpdater(appSettings.initialLearnRate)
        val evaluator = new MMLPInstanceEvaluator[BasicMMLPSgdUpdater](network)
        new OnlineOptimizer[MMLPFactor, MMLPWeights, MMLPLossGradient, BasicMMLPSgdUpdater](weights, evaluator, uu, appSettings)
      case "adam" =>
        val mom1 = network.generateZeroedLayout
        val mom2 = network.generateZeroedLayout
        val uu = new MMLPAdamUpdater(0.001f, 0.9f, 0.999f, mom1, mom2, maxNormArray=maxNormArray, l1Array = l1Array, l2Array = l2Array,
            composeSt = composeStrategy)
        val evaluator = new MMLPInstanceEvaluator[MMLPAdamUpdater](network)
        new OnlineOptimizer[MMLPFactor, MMLPWeights, MMLPLossGradient, MMLPAdamUpdater](weights, evaluator, uu, appSettings)
      case "lbfgs" => throw new RuntimeException("Batch training not supported without Spark")
      case a => throw new RuntimeException("Unrecognized online training method: " + a)
    }
  }

}
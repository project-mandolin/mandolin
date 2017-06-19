package org.mitre.mandolin.glp.local

import org.mitre.mandolin.glp._
import org.mitre.mandolin.optimize.local.LocalOnlineOptimizer

object LocalGLPOptimizer {
  
  def getLocalOptimizer(network: ANNetwork, method: String, lr: Float, numEpochs: Int) = {
    
  }
  
  def getLocalOptimizer(appSettings: MandolinMLPSettings, network: ANNetwork) = {
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
        val evaluator = new GLPInstanceEvaluator[GLPAdaGradUpdater](network)
        new LocalOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPAdaGradUpdater](weights, evaluator, updater, appSettings)
      case "adadelta" =>
        val prevUpdates = network.generateZeroedLayout
        val uu = new GLPAdaDeltaUpdater(sumSquared, prevUpdates, appSettings.epsilon, appSettings.rho,  
            compose = composeStrategy, maxNorm=appSettings.maxNorm)
        val evaluator = new GLPInstanceEvaluator[GLPAdaDeltaUpdater](network)
        new LocalOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPAdaDeltaUpdater](weights, evaluator, uu, appSettings)
      case "rmsprop" =>
        val sumSquared = network.generateZeroedLayout
        sumSquared set appSettings.initialLearnRate // set to the initial learning rate
        val updater = new GLPRMSPropUpdater(sumSquared, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array = l1Array, l2Array = l2Array,
            compose = composeStrategy)
        val evaluator = new GLPInstanceEvaluator[GLPRMSPropUpdater](network)
        new LocalOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPRMSPropUpdater](weights, evaluator, updater, appSettings)
      case "nasgd" => // Nesterov accelerated momentum-based SGD
        val momentum = network.generateZeroedLayout
        val uu = new GLPSgdUpdater(momentum, true, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array = l1Array, l2Array = l2Array,
            compose = composeStrategy)
        val evaluator = new GLPInstanceEvaluator[GLPSgdUpdater](network)
        new LocalOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPSgdUpdater](weights, evaluator, uu, appSettings) 
      case "sgd" =>
        val uu = new BasicGLPSgdUpdater(appSettings.initialLearnRate)
        val evaluator = new GLPInstanceEvaluator[BasicGLPSgdUpdater](network)
        new LocalOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, BasicGLPSgdUpdater](weights, evaluator, uu, appSettings)
      case "adam" =>
        val mom1 = network.generateZeroedLayout
        val mom2 = network.generateZeroedLayout
        val uu = new GLPAdamUpdater(0.001f, 0.9f, 0.999f, mom1, mom2, maxNormArray=maxNormArray, l1Array = l1Array, l2Array = l2Array,
            composeSt = composeStrategy)
        val evaluator = new GLPInstanceEvaluator[GLPAdamUpdater](network)
        new LocalOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPAdamUpdater](weights, evaluator, uu, appSettings)
      case "lbfgs" => throw new RuntimeException("Batch training not supported without Spark")
        /*
        val localBatchEvaluator = new GlpLocalBatchEvaluator(evaluator)
        val dim = weights.numWeights
        new LocalBatchOptimizer[GLPFactor, GLPWeights, GLPLossGradient](dim, weights, localBatchEvaluator, new Params())
        * 
        */
      case a => throw new RuntimeException("Unrecognized online training method: " + a)
    }
  }

}
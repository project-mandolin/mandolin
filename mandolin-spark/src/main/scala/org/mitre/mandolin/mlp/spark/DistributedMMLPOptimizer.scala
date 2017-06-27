package org.mitre.mandolin.mlp.spark

import org.mitre.mandolin.mlp._
import org.mitre.mandolin.optimize.spark.DistributedOnlineOptimizer
import org.mitre.mandolin.optimize.Updater
import org.apache.spark.SparkContext
import org.apache.spark.AccumulatorParam

object DistributedMMLPOptimizer {

  def getDistributedOptimizer(sc: SparkContext, appSettings: MandolinMLPSettings, network: ANNetwork) = {
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
    
    def getAccumulator[U <: Updater[MMLPWeights, MMLPLossGradient, U]] = new AccumulatorParam[U] {
      def zero(v: U) = {
            v.resetLearningRates(0.0f)
            v
          }
      def addInPlace(v1: U, v2: U) = v1 compose v2
    }
    
    appSettings.method match {
      case "adagrad" =>
        val sumSquared = network.generateZeroedLayout
        sumSquared set appSettings.initialLearnRate // set to the initial learning rate
        val updater = new MMLPAdaGradUpdater(sumSquared, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array=l1Array, l2Array=l2Array,
            compose = composeStrategy)
        val evaluator = new MMLPInstanceEvaluator[MMLPAdaGradUpdater](network)
        new DistributedOnlineOptimizer[MMLPFactor, MMLPWeights, MMLPLossGradient, MMLPAdaGradUpdater](sc, weights, evaluator, updater, appSettings)
      case "adadelta" =>
        val sumSquared = network.generateZeroedLayout
        val prevUpdates = network.generateZeroedLayout
        val up = new MMLPAdaDeltaUpdater(sumSquared, prevUpdates, appSettings.epsilon, appSettings.rho, compose = composeStrategy, maxNorm=appSettings.maxNorm)
        val evaluator = new MMLPInstanceEvaluator[MMLPAdaDeltaUpdater](network)
        new DistributedOnlineOptimizer[MMLPFactor, MMLPWeights, MMLPLossGradient, MMLPAdaDeltaUpdater](sc, weights, evaluator, up, appSettings)
      case "rmsprop" =>
        val sumSquared = network.generateZeroedLayout
        sumSquared set appSettings.initialLearnRate // set to the initial learning rate
        val updater = new MMLPRMSPropUpdater(sumSquared, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array=l1Array, l2Array=l2Array,
            compose = composeStrategy)
        val evaluator = new MMLPInstanceEvaluator[MMLPRMSPropUpdater](network)
        new DistributedOnlineOptimizer[MMLPFactor, MMLPWeights, MMLPLossGradient, MMLPRMSPropUpdater](sc, weights, evaluator, updater, appSettings)
      case "nasgd" => // Nesterov accelerated
        val momentum = network.generateZeroedLayout
        val uu = new MMLPSgdUpdater(momentum, true, appSettings.initialLearnRate, maxNormArray=maxNormArray, l1Array=l1Array, l2Array=l2Array,
            compose = composeStrategy)
        val evaluator = new MMLPInstanceEvaluator[MMLPSgdUpdater](network)
        new DistributedOnlineOptimizer[MMLPFactor, MMLPWeights, MMLPLossGradient, MMLPSgdUpdater](sc, weights, evaluator, uu, appSettings)
      case "adam" =>
        val mom1 = network.generateZeroedLayout
        val mom2 = network.generateZeroedLayout
        val uu = new MMLPAdamUpdater(0.001f, 0.9f, 0.999f, mom1, mom2, maxNormArray=maxNormArray, l1Array = l1Array, l2Array = l2Array,
            composeSt = composeStrategy)
        val evaluator = new MMLPInstanceEvaluator[MMLPAdamUpdater](network)
        new DistributedOnlineOptimizer[MMLPFactor, MMLPWeights, MMLPLossGradient, MMLPAdamUpdater](sc, weights, evaluator, uu, appSettings)
      case "sgd" =>
        val up = new BasicMMLPSgdUpdater(appSettings.initialLearnRate)
        val evaluator = new MMLPInstanceEvaluator[BasicMMLPSgdUpdater](network)
        new DistributedOnlineOptimizer[MMLPFactor, MMLPWeights, MMLPLossGradient, BasicMMLPSgdUpdater](sc, weights, evaluator, up, appSettings)
      case a => throw new RuntimeException("Unrecognized online training method: " + a)
    }
  }

}
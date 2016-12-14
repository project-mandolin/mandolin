package org.mitre.mandolin.mselect

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.mitre.mandolin.glp.GLPModelSettings
import org.mitre.mandolin.glp.local.{LocalGLPOptimizer, LocalProcessor}
import org.mitre.mandolin.predict.local.{LocalEvalDecoder, LocalTrainer}
import org.mitre.mandolin.util.LocalIOAssistant

abstract class LearnerInstance extends LocalProcessor {
  def train(sc: SparkContext, trainBC: Broadcast[Vector[String]], testBC: Broadcast[Vector[String]]): Double
}

abstract class LearnerFactory {
  def getLearnerInstance(config: ModelConfig): LearnerInstance
}

class MandolinLogisticRegressionInstance(appSettings: GLPModelSettings, config: ModelConfig) extends LearnerInstance {
  def train(sc: SparkContext, trainBC: Broadcast[Vector[String]], testBC: Broadcast[Vector[String]]) {
    val io = new LocalIOAssistant
    val components = getComponentsViaSettings(appSettings, io)
    val fe = components.featureExtractor
    val pr = components.predictor
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(appSettings, components.ann)
    val trainLines = trainBC.value
    val testLines = testBC.value
    val trainer = new LocalTrainer(fe, optimizer)
    val evPr = new LocalEvalDecoder(trainer.fe, pr)
    val trainVectors = trainer.extractFeatures(trainLines)
    val testVectors  = evPr.extractFeatures(testLines)
    for (i <- 1 to appSettings.numEpochs) {
      val t = System.nanoTime()
      val (weights,trainLoss) = trainer.retrainWeights(trainVectors, 1)
      val confusion = evPr.evalUnits(testVectors, weights)
      val confMat = confusion.getMatrix
      val acc = confMat.getAccuracy
    }
    val (weights,trainLoss) = trainer.retrainWeights(trainVectors, 1)
    val confusion = evPr.evalUnits(testVectors, weights)
    val confMat = confusion.getMatrix
    val acc = confMat.getAccuracy
    acc
  }
}

class MandolinLogisticRegressionFactory extends LearnerFactory {
  override def getLearnerInstance(config: ModelConfig): LearnerInstance = {

    val cats: Vector[Option[Vector[String]]] = config.categoricalMetaParamSet.map { param => {
      val paramValue: String = param.getValue.s
      param.getName match {
        case "method" => Some(Vector("--mandolin.trainer.optimizer.method", paramValue))
        case _ => None
      }
    }
    }

    val reals: Vector[Option[Vector[String]]] = config.realMetaParamSet.map { param => {
      val paramValue: RealValue = param.getValue
      param.getName match {
        case "lr" => Some(Vector("--mandolin.trainer.optimizer.initial-learning-rate", paramValue.toString))
        case _ => None
      }
    }
    }

    val args: Vector[String] = (cats ++ reals) filter { opt => opt.isDefined } map { opt => opt.get } flatten
    val settings = new GLPModelSettings(args.toArray)
    new MandolinLogisticRegressionInstance(settings, config)
  }
}
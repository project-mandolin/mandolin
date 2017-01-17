package org.mitre.mandolin.mselect

import org.apache.spark.broadcast.Broadcast
import org.mitre.mandolin.glp.{CategoricalGLPPredictor, ANNetwork, GLPFactor, GLPWeights, GLPComponentSet, GLPModelSettings}
import org.mitre.mandolin.glp.local.{LocalGLPOptimizer, LocalProcessor}
import org.mitre.mandolin.predict.local.{LocalEvalDecoder, NonExtractingEvalDecoder, LocalTrainer}
import org.mitre.mandolin.predict.DiscreteConfusion
import org.mitre.mandolin.util.LocalIOAssistant

abstract class LearnerInstance[T] extends LocalProcessor {
  def train(trainBC: Broadcast[Vector[T]], testBC: Broadcast[Vector[T]]): Double
  def train(trainBC: Vector[T], testBC: Vector[T]): Double
}

abstract class LearnerFactory[T] {
  def getLearnerInstance(config: ModelConfig): LearnerInstance[T]
}

class MandolinLogisticRegressionInstance(appSettings: GLPModelSettings, config: ModelConfig, nn: ANNetwork) 
extends LearnerInstance[GLPFactor] with Serializable {

  
  def train(train: Vector[GLPFactor], test: Vector[GLPFactor]) : Double = {
    val io = new LocalIOAssistant
    val lp = new LocalProcessor

    val optimizer = LocalGLPOptimizer.getLocalOptimizer(appSettings, nn)

    val predictor = new CategoricalGLPPredictor(nn, true)

    val trainer = new LocalTrainer(optimizer)
    val evPr = new NonExtractingEvalDecoder[GLPFactor,GLPWeights,Int,DiscreteConfusion](predictor)
    val (weights, trainLoss) = trainer.retrainWeights(train, appSettings.numEpochs)
    val confusion = evPr.evalUnits(test, weights)
    val confMat = confusion.getMatrix
    val acc = confMat.getAccuracy
    acc
  }

  def train(trainBC: Broadcast[Vector[GLPFactor]], testBC: Broadcast[Vector[GLPFactor]]): Double = {
    train(trainBC.value, testBC.value)    
  }
}

class MandolinLogisticRegressionFactory extends LearnerFactory[GLPFactor] {
  def getLearnerInstance(config: ModelConfig): LearnerInstance[GLPFactor] = {

    val cats: Vector[Option[String]] = config.categoricalMetaParamSet.map { param => {
      val paramValue: String = param.getValue.s
      param.getName match {
        case "method" => Some("mandolin.trainer.optimizer.method=" + paramValue)
        case "numTrainerThreads" => Some("mandolin.trainer.threads=" + paramValue)
        case _ => None
      }
    }
    }

    val reals: Vector[Option[String]] = config.realMetaParamSet.map { param => {
      val paramValue: RealValue = param.getValue
      param.getName match {
        case "lr" => Some("mandolin.trainer.optimizer.initial-learning-rate=" + paramValue.v)
        case _ => None
      }
    }
    }

    val args: Vector[String] = (cats ++ reals) filter { opt => opt.isDefined } map { opt => opt.get }
    // XXX - the above can be simplified with new way to setting up model settings
    val settings = new GLPModelSettings(args.toArray)
    new MandolinLogisticRegressionInstance(settings, config, config.mSpec)
  }
}
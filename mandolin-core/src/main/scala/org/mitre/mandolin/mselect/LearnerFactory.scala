package org.mitre.mandolin.mselect

import org.mitre.mandolin.glp.{CategoricalGLPPredictor, ANNetwork, GLPFactor, GLPWeights, GLPComponentSet, GLPModelSettings}
import org.mitre.mandolin.glp.local.{LocalGLPOptimizer, LocalProcessor}
import org.mitre.mandolin.predict.local.{LocalEvalDecoder, NonExtractingEvalDecoder, LocalTrainer}
import org.mitre.mandolin.predict.DiscreteConfusion
import org.mitre.mandolin.util.LocalIOAssistant

abstract class LearnerInstance[T] extends LocalProcessor {
  def train(trainBC: Vector[T], testBC: Vector[T]): Double
}

abstract class LearnerFactory[T] {
  def getLearnerInstance(config: ModelConfig): LearnerInstance[T]
}

class MandolinLogisticRegressionInstance(appSettings: GLPModelSettings, config: ModelConfig, nn: ANNetwork) 
extends LearnerInstance[GLPFactor] with Serializable {

  
  def train(train: Vector[GLPFactor], test: Vector[GLPFactor]) : Double = {
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(appSettings, nn)

    val predictor = new CategoricalGLPPredictor(nn, true)

    val trainer = new LocalTrainer(optimizer)
    val evPr = new NonExtractingEvalDecoder[GLPFactor,GLPWeights,Int,DiscreteConfusion](predictor)
    val (weights, trainLoss) = trainer.retrainWeights(train, appSettings.numEpochs)
    val confusion = evPr.evalWithoutExtraction(test, weights)
    val acc = confusion.getAccuracy
    acc
  }

}

class MandolinLogisticRegressionFactory extends LearnerFactory[GLPFactor] {
  def getLearnerInstance(config: ModelConfig): LearnerInstance[GLPFactor] = {

    val cats: List[(String, Any)] = config.categoricalMetaParamSet.foldLeft(Nil:List[(String,Any)]) {case (ac,v) =>
      v.getName match {
        case "method" =>            ("mandolin.trainer.optimizer.method", v.getValue.s) :: ac
        case "numTrainerThreads" => ("mandolin.trainer.threads", v.getValue.s) :: ac
        case _ => ac
      }
    }

    val reals: List[(String,Any)] = config.realMetaParamSet.foldLeft(Nil:List[(String,Any)]) { case (ac,v) => 
      val paramValue: RealValue = v.getValue
      v.getName match {
        case "lr" => ("mandolin.trainer.optimizer.initial-learning-rate", paramValue.v) :: ac
        case _ => ac
      }    
    }

    val allParams : Seq[(String,Any)] = (cats ++ reals) toSeq 
    val settings = new GLPModelSettings().withSets(allParams)
    val annCopy = config.mSpec.copy() // need to copy the ann so that separate threads aren't overwriting outputs/derivatives/etc.
    new MandolinLogisticRegressionInstance(settings, config, annCopy)
  }
}
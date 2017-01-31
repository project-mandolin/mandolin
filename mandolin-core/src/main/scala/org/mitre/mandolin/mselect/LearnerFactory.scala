package org.mitre.mandolin.mselect

import org.mitre.mandolin.glp.{CategoricalGLPPredictor, ANNetwork, GLPFactor, GLPWeights, GLPComponentSet, GLPModelSettings}
import org.mitre.mandolin.glp.local.{LocalGLPOptimizer, LocalProcessor}
import org.mitre.mandolin.predict.local.{LocalEvalDecoder, NonExtractingEvalDecoder, LocalTrainer}
import org.mitre.mandolin.predict.DiscreteConfusion
import org.mitre.mandolin.util.LocalIOAssistant

import scala.collection.immutable.IndexedSeq
import scala.collection.mutable

trait LearnerInstance[T] extends LocalProcessor {
  def train(train: Vector[T], test: Vector[T]): Double
}

trait LearnerFactory[T] {
  def getLearnerInstance(config: ModelConfig): LearnerInstance[T]
  def getModelSpaceBuilder : ModelSpaceBuilder
}

trait ModelSpaceBuilder {
  val reals = new mutable.MutableList[RealMetaParameter]
  val cats = new mutable.MutableList[CategoricalMetaParameter]

  def withMetaParam(realMP: RealMetaParameter) = {
    reals += realMP
    this
  }

  def withMetaParam(catMP: CategoricalMetaParameter) = {
    cats += catMP
    this
  }

  def build() : ModelSpace = {
    new ModelSpace(reals.toVector, cats.toVector)
  }
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

//trait MandolinLogisticRegressionFactory extends LearnerFactory[GLPFactor]
object MandolinLogisticRegressionFactory extends LearnerFactory[GLPFactor] {
  
  // XXX - factory should know the number of inputs/outputs for the MLP
  // XXX - specification should come as a meta-parameter in config
  // XXX - ANN instance created as part of instantiating a config (rather than simple copy of static ANN below)

  class MandolinLogisticRegressionModelSpaceBuilder extends ModelSpaceBuilder {
    def defineInitialLearningRates(start: Double, end: Double): ModelSpaceBuilder = {
      withMetaParam(new RealMetaParameter("lr", new RealSet(start, end)))
    }

    def defineOptimizerMethods(methods: String*) = {
      withMetaParam(new CategoricalMetaParameter("method", new CategoricalSet(methods.toVector)))
    }

    def defineTrainerThreads(numTrainerThreads : Int) = {
      withMetaParam(new CategoricalMetaParameter("numTrainerThreads", new CategoricalSet(Vector(numTrainerThreads.toString))))
    }
  }

  override def getModelSpaceBuilder() : MandolinLogisticRegressionModelSpaceBuilder = {
    new MandolinLogisticRegressionModelSpaceBuilder
  }

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
package org.mitre.mandolin.mselect

import org.apache.spark.broadcast.Broadcast
import org.mitre.mandolin.glp.{GLPFactor, GLPComponentSet, GLPModelSettings}
import org.mitre.mandolin.glp.local.{LocalGLPOptimizer, LocalProcessor}
import org.mitre.mandolin.predict.local.{LocalEvalDecoder, LocalTrainer}
import org.mitre.mandolin.util.LocalIOAssistant

abstract class LearnerInstance[T] extends LocalProcessor {
  def train(trainBC: Broadcast[Vector[T]], testBC: Broadcast[Vector[T]]): Double
  def train(trainBC: Vector[T], testBC: Vector[T]): Double
}

abstract class LearnerFactory[T] {
  def getLearnerInstance(config: ModelConfig): LearnerInstance[T]
}

class MandolinLogisticRegressionInstance(appSettings: GLPModelSettings, config: ModelConfig) extends LearnerInstance[GLPFactor] with Serializable {

  def trainFeats(trainBC: Broadcast[Vector[GLPFactor]], testBC: Broadcast[Vector[GLPFactor]]): Double = {
    val io = new LocalIOAssistant
    val lp = new LocalProcessor
    val components: GLPComponentSet = lp.getComponentsViaSettings(appSettings, io)
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(appSettings, components.ann)

    val fe = components.featureExtractor
    val pr = components.predictor

    val trainer = new LocalTrainer(fe, optimizer)
    val evPr = new LocalEvalDecoder(trainer.getFe, pr)
    val trainVectors = trainBC.value
    val testVectors = testBC.value
    for (i <- 1 to appSettings.numEpochs) {
      val (weights, trainLoss) = trainer.retrainWeights(trainVectors, 1)
      val confusion = evPr.evalUnits(testVectors, weights)
      val confMat = confusion.getMatrix
      val acc = confMat.getAccuracy
    }
    val (weights, trainLoss) = trainer.retrainWeights(trainVectors, 1)
    val confusion = evPr.evalUnits(testVectors, weights)
    val confMat = confusion.getMatrix
    val acc = confMat.getAccuracy
    acc
  }
  
  
  def train(train: Vector[GLPFactor], test: Vector[GLPFactor]) : Double = {
    val io = new LocalIOAssistant
    val lp = new LocalProcessor
    val components: GLPComponentSet = lp.getComponentsViaSettings(appSettings, io)
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(appSettings, components.ann)

    val fe = components.featureExtractor
    val pr = components.predictor

    val trainer = new LocalTrainer(fe, optimizer)
    val evPr = new LocalEvalDecoder(trainer.getFe, pr)
    //val trainVectors: Vector[GLPFactor] = trainer.extractFeatures(train)
    //val testVectors = evPr.extractFeatures(test)
    for (i <- 1 to appSettings.numEpochs) {
      val (weights, trainLoss) = trainer.retrainWeights(train, 1)
      val confusion = evPr.evalUnits(test, weights)
      val confMat = confusion.getMatrix
      val acc = confMat.getAccuracy
    }
    val (weights, trainLoss) = trainer.retrainWeights(train, 1)
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
    val fixme = args :+ "mandolin.trainer.train-file=/nfshome/jkraunelis/test.vectors" :+ "mandolin.trainer.test-file=/nfshome/jkraunelis/test.vectors"// :+ "mandolin.trainer.num-hashed-features=1000000" :+ "mandolin.trainer.use-random-features=true"
    val settings = new GLPModelSettings(fixme.toArray)
    new MandolinLogisticRegressionInstance(settings, config)
  }
}
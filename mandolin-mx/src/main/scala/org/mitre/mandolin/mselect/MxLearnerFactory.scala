package org.mitre.mandolin.mselect

import org.mitre.mandolin.glp.GLPFactor
import org.mitre.mandolin.glp.{GLPModelSettings, LType, SparseInputLType, InputLType, SoftMaxLType}
import org.mitre.mandolin.mx.{MxModelSettings, MxNetSetup}

import ml.dmlc.mxnet.{Uniform, Xavier, Context, Shape}
import ml.dmlc.mxnet.optimizer._
import org.mitre.mandolin.mx.{MxNetOptimizer, MxNetWeights, MxNetEvaluator, SymbolBuilder, GLPFactorIter}

import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

import org.slf4j.LoggerFactory


class MxModelSpaceBuilder(ms: Option[ModelSpace]) extends ModelSpaceBuilder {

  def this(m: ModelSpace) = this(Some(m))

  def this() = this(None)

  // initialize with modelspace
  ms foreach { ms =>
    ms.catMPs foreach withMetaParam
    ms.realMPs foreach withMetaParam
    ms.intMPs foreach withMetaParam

  }

  def build(idim: Int, odim: Int, sparse: Boolean, appSettings: Option[MxModelSettings]) : ModelSpace = {    
    val budget = appSettings match {case Some(m) => m.numEpochs case None => -1}
    // Pull out important parameters to preserve here and pass into model space
    val appConfig = appSettings map {a => a.config.root.render()}
    new ModelSpace(reals.toVector, cats.toVector, ints.toVector, idim, odim, appConfig, budget)    
  }  
}

trait MxLearnerBuilderHelper {
  def setupSettings(config: ModelConfig): MxModelSettings = {
    val budget = config.budget // number of iterations, typically
    val cats: List[(String, Any)] = config.categoricalMetaParamSet.toList map { cm => (cm.getName, cm.getValue.s) }
    val reals: List[(String, Any)] = config.realMetaParamSet.toList map { cm => (cm.getName, cm.getValue.v) }
    val ints: List[(String, Any)] = config.intMetaParamSet.toList map { cm => (cm.getName, cm.getValue.v) }
    val setBudget: List[(String, Any)] = if (budget > 0) List(("mandolin.trainer.num-epochs", budget)) else Nil
    val allParams: Seq[(String, Any)] = (cats ++ reals ++ ints ++ setBudget) toSeq
    val completeParams = allParams
    val mxsets = config.serializedSettings match {
      case Some(s) => new MxModelSettings(s)
      case None => new MxModelSettings()
    }
    mxsets.withSets(completeParams)
  }
}

class MxModelInstance(appSettings: MxModelSettings, nfs: Int, modelId: Int, startFrom: Int)
extends LearnerInstance[GLPFactor] with MxNetSetup {
    
  def train(trVecs: Vector[GLPFactor], tstVecs: Option[Vector[GLPFactor]]) : Double = {
    val devices = getDeviceArray(appSettings)
    val sym = (new SymbolBuilder).symbolFromSpec(appSettings.config)
    val shape = Shape(nfs)
    val trIter = new GLPFactorIter(trVecs.toIterator, shape, appSettings.miniBatchSize)
    val tstIter = tstVecs match {
      case Some(tst) => new GLPFactorIter(tst.toIterator, shape, appSettings.miniBatchSize)
      case None => trIter
    }
    val lr = appSettings.initialLearnRate
    val opt = getOptimizer(appSettings)
    val updater = new MxNetOptimizer(opt)
    val weights = new MxNetWeights(1.0f)
    val initializer = appSettings.mxInitializer match {
      case "xavier" => new Xavier(rndType = "gaussian", factorType = "in", magnitude = 1.8f)
      case _ => new Uniform(0.01f)
    }
    val modelPrefix = appSettings.modelFile match {
      case Some(s) => Some(s + "-id-" + modelId)
      case None => None
    }
    val evaluator = new MxNetEvaluator(sym, devices, shape, appSettings.miniBatchSize, initializer, modelPrefix, -1)
    val lg = evaluator.evaluateTrainingMiniBatch(trIter, tstIter, weights, updater, appSettings.numEpochs, startFrom)
    sym.dispose()
    lg.loss
  }
    
}

object MxModelInstance extends MxLearnerBuilderHelper {
  def apply(config: ModelConfig): MxModelInstance = {
    val settings = setupSettings(config)
    new MxModelInstance(settings, config.inDim, config.id, config.src)
  }
}

class FileSystemImgMxModelInstance(appSettings: MxModelSettings, nfs: Int, modelId: Int, startFrom: Int)
  extends LearnerInstance[java.io.File] with MxNetSetup {
  val log = LoggerFactory.getLogger(getClass)

  def train(trData: Vector[java.io.File], tstData: Option[Vector[java.io.File]]) : Double = {
    val devices = getDeviceArray(appSettings)
    val sym = (new SymbolBuilder).symbolFromSpec(appSettings.config)
    val (shape, trIter, tstIter) = if (appSettings.inputType equals "mnist") {
      val yd = appSettings.ydim // if this is greater than 0 assume data is shaped, otherwise flat with xdim providing dimensions
      val s = if (yd > 0) Shape(appSettings.channels, appSettings.xdim, yd) else Shape(appSettings.xdim)
      (s, getMNISTTrainIO(appSettings, s), getMNISTTestIO(appSettings, s))
    } else {
      val shape = Shape(appSettings.channels, appSettings.xdim, appSettings.ydim)
      (shape, getTrainIO(appSettings, shape), getTestIO(appSettings, shape))
    }
    val lr = appSettings.initialLearnRate
    val opt = getOptimizer(appSettings)
    val updater = new MxNetOptimizer(opt)
    val weights = new MxNetWeights(1.0f)
    val init = appSettings.mxInitializer match {
      case "xavier" => new Xavier(rndType = "gaussian", factorType = "in", magnitude = 1.8f)
      case _ => new Uniform(0.01f)
    }

    val modelPrefix = appSettings.modelFile match {
      case Some(s) => Some(s + "-id-" + modelId)
      case None => None
    }
    log.info("^^^^^^ => Evaluating model via mxnet with prefix = " + modelPrefix)
    val evaluator = new MxNetEvaluator(sym, devices, shape, appSettings.miniBatchSize, init, modelPrefix, -1)
    val lg = evaluator.evaluateTrainingMiniBatch(trIter, tstIter, weights, updater, appSettings.numEpochs, startFrom)
    sym.dispose()
    lg.loss
  }
  
  
}

object FileSystemImgMxModelInstance extends MxLearnerBuilderHelper {
  def apply(config: ModelConfig): FileSystemImgMxModelInstance = {
    val settings = setupSettings(config)
    val startFrom = config.src
    new FileSystemImgMxModelInstance(settings, config.inDim, config.id, startFrom)
  }
}

class FileSystemMxModelEvaluator(trData: java.io.File, tstData: java.io.File) extends ModelEvaluator with Serializable {
  val log = LoggerFactory.getLogger(getClass)

  def evaluate(c: ModelConfig, generation: Int): (Double, Long) = {
    val learner = FileSystemImgMxModelInstance(c)
    val startTime = System.currentTimeMillis()
    val acc = learner.train(Vector(trData), Some(Vector(tstData)))
    val endTime = System.currentTimeMillis()
    (acc, endTime - startTime)
  }

  override def cancel(generation: Int): Unit = null
}


class LocalMxModelEvaluator(trData: Vector[GLPFactor], tstData: Option[Vector[GLPFactor]]) extends ModelEvaluator with Serializable {

  def evaluate(c: ModelConfig, generation: Int): (Double, Long) = {
    val learner = MxModelInstance(c)
    val startTime = System.currentTimeMillis()
    val acc = learner.train(trData, tstData)
    val endTime = System.currentTimeMillis()
    (acc, endTime - startTime)
  }
  override def cancel(generation: Int): Unit = null
}
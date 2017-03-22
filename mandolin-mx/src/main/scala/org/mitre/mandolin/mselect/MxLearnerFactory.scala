package org.mitre.mandolin.mselect

import org.mitre.mandolin.glp.GLPFactor
import org.mitre.mandolin.glp.{GLPModelSettings, LType, SparseInputLType, InputLType,SoftMaxLType}
import org.mitre.mandolin.mx.{MxModelSettings, MxNetSetup}

import ml.dmlc.mxnet.{Context, Shape}
import ml.dmlc.mxnet.optimizer._
import org.mitre.mandolin.mx.{ MxNetOptimizer, MxNetWeights, MxNetEvaluator, SymbolBuilder, GLPFactorIter}
  
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

import org.slf4j.LoggerFactory


class MxModelSpaceBuilder(ms: Option[ModelSpace]) extends ModelSpaceBuilder {

  def this(m: ModelSpace) = this(Some(m))
  def this() = this(None)
  
  // initialize with modelspace
  ms foreach {ms => 
    ms.catMPs foreach withMetaParam
    ms.realMPs foreach withMetaParam
    ms.intMPs foreach withMetaParam
    ms.topoMPs foreach withMetaParam
  }
       
  def build(idim: Int, odim: Int, sparse: Boolean, appSettings: Option[MxModelSettings]) : ModelSpace = {    
    val it = if (sparse) LType(SparseInputLType, idim) else LType(InputLType, odim)
    // Pull out important parameters to preserve here and pass into model space
    val appConfig = appSettings map {a => a.config.root.render()}
    new ModelSpace(reals.toVector, cats.toVector, ints.toVector, topo, it, LType(SoftMaxLType, odim), idim, odim, appConfig)    
  }  
}

trait MxLearnerBuilderHelper {
  def setupSettings(config: ModelConfig) : MxModelSettings = {
    val cats: List[(String,Any)] = config.categoricalMetaParamSet.toList map {cm => (cm.getName,cm.getValue.s)}
    val reals : List[(String,Any)] = config.realMetaParamSet.toList map {cm => (cm.getName,cm.getValue.v)}
    val ints : List[(String,Any)] = config.intMetaParamSet.toList map {cm => (cm.getName, cm.getValue.v)}    
    val allParams : Seq[(String,Any)] = (cats ++ reals ++ ints) toSeq 
    val completeParams = allParams     
    val mxsets = config.serializedSettings match {case Some(s) => new MxModelSettings(s) case None => new MxModelSettings() }
    mxsets.withSets(completeParams)  
  }  
}

class MxModelInstance(appSettings: MxModelSettings, nfs: Int) extends LearnerInstance[GLPFactor] with MxNetSetup {
  
  val log = LoggerFactory.getLogger(getClass)
  
  def train(trVecs: Vector[GLPFactor], tstVecs: Vector[GLPFactor]) : Double = {
    log.info("Initiating training ...")
    val devices = getDeviceArray(appSettings)
    val sym     = (new SymbolBuilder).symbolFromSpec(appSettings.config)    
    val shape = Shape(nfs)
    val trIter = new GLPFactorIter(trVecs.toIterator, shape, appSettings.miniBatchSize)
    val tstIter = new GLPFactorIter(tstVecs.toIterator, shape, appSettings.miniBatchSize)
    val lr = appSettings.initialLearnRate
    val opt = getOptimizer(appSettings)
    val updater = new MxNetOptimizer(opt)
    val weights = new MxNetWeights(1.0f)    
    val evaluator = new MxNetEvaluator(sym, devices, shape, appSettings.miniBatchSize, appSettings.modelFile, appSettings.saveFreq)
    val lg = evaluator.evaluateTrainingMiniBatch(trIter, tstIter, weights, updater, appSettings.numEpochs)
    lg.loss
  }
}

object MxModelInstance extends MxLearnerBuilderHelper {
  def apply(config: ModelConfig) : MxModelInstance = {
    val settings = setupSettings(config)
    new MxModelInstance(settings, config.inDim)
  }
}

class FileSystemImgMxModelInstance(appSettings: MxModelSettings, nfs: Int) extends LearnerInstance[java.io.File] with MxNetSetup {
  val log = LoggerFactory.getLogger(getClass)
  
  def train(trData: Vector[java.io.File], tstData: Vector[java.io.File]) : Double = {
    val devices = getDeviceArray(appSettings)
    val sym     = (new SymbolBuilder).symbolFromSpec(appSettings.config)
    val shape   = Shape(appSettings.channels, appSettings.xdim, appSettings.ydim)
    val trIter = getTrainIO(appSettings, shape)
    val tstIter = getTestIO(appSettings, shape)
    val lr = appSettings.initialLearnRate
    val opt = getOptimizer(appSettings)
    val updater = new MxNetOptimizer(opt)
    val weights = new MxNetWeights(1.0f)    
    val evaluator = new MxNetEvaluator(sym, devices, shape, appSettings.miniBatchSize, appSettings.modelFile, appSettings.saveFreq)
    val lg = evaluator.evaluateTrainingMiniBatch(trIter, tstIter, weights, updater, appSettings.numEpochs)
    sym.dispose()
    lg.loss
  }
}
object FileSystemImgMxModelInstance extends MxLearnerBuilderHelper {
  def apply(config: ModelConfig) : FileSystemImgMxModelInstance = {
    val settings = setupSettings(config)
    new FileSystemImgMxModelInstance(settings, config.inDim)
  }
}

class FileSystemMxModelEvaluator(trData: java.io.File, tstData: java.io.File) extends ModelEvaluator with Serializable {
  val log = LoggerFactory.getLogger(getClass)
  
  def evaluate(c: Seq[ModelConfig]) : Seq[Double] = {
    log.info("Initiating evaluation with FileSystemMxModelEvaluator ... ")
    val cvec = c.toList.par
    cvec.tasksupport_=(new ForkJoinTaskSupport(new ForkJoinPool(cvec.length)))
    val accuracies = cvec map {config =>
      val learner = FileSystemImgMxModelInstance(config)
      val acc = learner.train(Vector(trData), Vector(tstData))
      acc
    }
    accuracies.seq  
  }
}


class LocalMxModelEvaluator(trData: Vector[GLPFactor], tstData: Vector[GLPFactor]) extends ModelEvaluator with Serializable {

  def evaluate(c: Seq[ModelConfig]): Seq[Double] = {
    val cvec = c.toList.par
    cvec.tasksupport_=(new ForkJoinTaskSupport(new ForkJoinPool(cvec.length)))
    val accuracies = cvec map {config =>
      val learner = MxModelInstance(config)
      val acc = learner.train(trData, tstData)
      acc
    }
    accuracies.seq
  }
}
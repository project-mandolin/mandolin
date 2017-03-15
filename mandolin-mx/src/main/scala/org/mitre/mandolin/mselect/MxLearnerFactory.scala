package org.mitre.mandolin.mselect

import org.mitre.mandolin.glp.GLPFactor
import org.mitre.mandolin.glp.{GLPModelSettings, LType, SparseInputLType, InputLType,SoftMaxLType}
import org.mitre.mandolin.mx.{MxModelSettings, MxNetSetup}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

import org.slf4j.LoggerFactory

trait MxModelSpaceBuilder extends ModelSpaceBuilder {
  
  def build(idim: Int, odim: Int, sparse: Boolean, appSettings: Option[MxModelSettings]) : ModelSpace = {    
    val it = if (sparse) LType(SparseInputLType, idim) else LType(InputLType, odim)
    // Pull out important parameters to preserve here and pass into model space
    val appConfig = appSettings map {a => a.config.root.render()}
    new ModelSpace(reals.toVector, cats.toVector, ints.toVector, topo, it, LType(SoftMaxLType, odim), idim, odim, appConfig)    
  }
}

object MxLearnerFactory extends LearnerFactory [GLPFactor]{
  
  val log = LoggerFactory.getLogger(getClass)

  class GenericModelSpaceBuilder extends MxModelSpaceBuilder {
    
    def withRealMetaParams(rs: Vector[RealMetaParameter]) = rs foreach withMetaParam 
    def withCategoricalMetaParams(cats: Vector[CategoricalMetaParameter]) = cats foreach withMetaParam
    def withIntegerMetaParams(ints: Vector[IntegerMetaParameter]) = ints foreach withMetaParam
    def withTopologyMetaParam(topo: TopologySpaceMetaParameter) = withMetaParam(topo)  
  }
  
  override def getModelSpaceBuilder() : GenericModelSpaceBuilder = {
    new GenericModelSpaceBuilder
  }
  
  def getModelSpaceBuilder(ms: ModelSpace) : GenericModelSpaceBuilder = {
    val mm = new GenericModelSpaceBuilder
    mm.withCategoricalMetaParams(ms.catMPs)
    mm.withRealMetaParams(ms.realMPs)
    mm.withIntegerMetaParams(ms.intMPs)
    ms.topoMPs foreach {ms => mm.withTopologyMetaParam(ms) }
    mm
  }
  
  def getLearnerInstance(config: ModelConfig) : LearnerInstance[GLPFactor] = {
    val cats: List[(String,Any)] = config.categoricalMetaParamSet.toList map {cm => (cm.getName,cm.getValue.s)}
    val reals : List[(String,Any)] = config.realMetaParamSet.toList map {cm => (cm.getName,cm.getValue.v)}
    val ints : List[(String,Any)] = config.intMetaParamSet.toList map {cm => (cm.getName, cm.getValue.v)}    
    val allParams : Seq[(String,Any)] = (cats ++ reals ++ ints) toSeq 
    val completeParams = allParams     
    val mxsets = config.serializedSettings match {case Some(s) => new MxModelSettings(s) case None => new MxModelSettings() }
    val settings = mxsets.withSets(completeParams)
    new MxModelInstance(settings, config.inDim)
  }
}


class MxModelInstance(appSettings: MxModelSettings, nfs: Int) extends LearnerInstance[GLPFactor] with MxNetSetup {
  import ml.dmlc.mxnet.{Context, Shape}
  import ml.dmlc.mxnet.optimizer._
  import org.mitre.mandolin.mx.{ MxNetOptimizer, MxNetWeights, MxNetEvaluator, SymbolBuilder, GLPFactorIter}
  
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

class LocalMxModelEvaluator(trData: Vector[GLPFactor], tstData: Vector[GLPFactor]) extends ModelEvaluator with Serializable {
  import org.slf4j.LoggerFactory
  
  val log = LoggerFactory.getLogger(getClass)

  override def evaluate(c: Seq[ModelConfig]): Seq[Double] = {
    val cvec = c.toList.par
    cvec.tasksupport_=(new ForkJoinTaskSupport(new ForkJoinPool(cvec.length)))
    val accuracies = cvec map {config =>
      val learner = MxLearnerFactory.getLearnerInstance(config)
      val acc = learner.train(trData, tstData)
      acc
    }
    accuracies.seq
  }
}
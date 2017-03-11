package org.mitre.mandolin.mselect

import org.mitre.mandolin.glp.GLPFactor
import org.mitre.mandolin.glp.{GLPModelSettings, LType, SparseInputLType, InputLType,SoftMaxLType}
import org.mitre.mandolin.mx.MxModelSettings
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

import org.slf4j.LoggerFactory

trait MxModelSpaceBuilder extends ModelSpaceBuilder {
  
  def build(idim: Int, odim: Int, sparse: Boolean, appSettings: Option[MxModelSettings]) : ModelSpace = {    
    val it = if (sparse) LType(SparseInputLType, idim) else LType(InputLType, odim)
    // Pull out important parameters to preserve here and pass into model space
    val opts : Option[Seq[(String,Any)]] = appSettings map { a =>
      Seq(
          ("mandolin.mx.input-type",a.inputType),
          ("mandolin.mx.num-classes", a.numberOfClasses),
          ("mandolin.mx.gpus", a.gpus),
          ("mandolin.mx.cpus", a.cpus),
          ("mandolin.mx.save-freq", a.saveFreq),
          ("mandolin.mx.train.initial-learning-rate", a.mxInitialLearnRate),
          ("mandolin.mx.train.rescale-gradient", a.mxRescaleGrad),
          ("mandolin.mx.train.momentum", a.mxMomentum), 
          ("mandolin.mx.train.gradient-clip", a.mxGradClip),
          ("mandolin.mx.train.rho", a.mxRho),
          ("mandolin.mx.train.optimizer", a.mxOptimizer),
          ("mandolin.mx.img.channels", a.channels),
          ("mandoinn.mx.img.xdim", a.xdim),
          ("mandolin.mx.img.ydim", a.ydim),
          ("mandolin.mx.img.mean-image", a.meanImgFile),
          ("mandolin.mx.img.preprocess-threads", a.preProcThreads),
          ("mandolin.mx.specification",a.mxSpecification)
          )
    } 
    
    new ModelSpace(reals.toVector, cats.toVector, ints.toVector, topo, it, LType(SoftMaxLType, odim), idim, odim, opts)    
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
    log.info("Getting learner instance set up...")
    val cats: List[(String,Any)] = config.categoricalMetaParamSet.toList map {cm => (cm.getName,cm.getValue.s)}
    val reals : List[(String,Any)] = config.realMetaParamSet.toList map {cm => (cm.getName,cm.getValue.v)}
    val ints : List[(String,Any)] = config.intMetaParamSet.toList map {cm => (cm.getName, cm.getValue.v)}
    
    //val mspecValued = config.topoMPs map {ms => ms.getValue.v.s map {l => l.drawRandomValue.getValue} map {vl => getSpec(vl)}}
    
    
    //val fullSpec : Vector[LType] = Vector(config.inLType) ++  hiddenLayers ++ Vector(config.outLType)
    //val net = ANNetwork(fullSpec, config.inDim, config.outDim)
    val allParams : Seq[(String,Any)] = (cats ++ reals ++ ints) toSeq 
    val completeParams = allParams ++ config.fixedSettingValues  // add in fixed settings
    log.info("Getting mx model params...")
    val mxsets = new MxModelSettings()
    log.info("About to call withSets...")
    val settings1 = mxsets.withSets(completeParams)
    val settings = settings1.asInstanceOf[MxModelSettings]
    log.info("Getting mx model instance")
    println("Model instance")
    System.out.println
    System.out.flush()
    System.err.flush()
    new MxModelInstance(settings, config.inDim)
  }
}


class MxModelInstance(appSettings: MxModelSettings, nfs: Int) extends LearnerInstance[GLPFactor] {
  import ml.dmlc.mxnet.{Context, Shape}
  import ml.dmlc.mxnet.optimizer._
  import org.mitre.mandolin.mx.{ MxNetOptimizer, MxNetWeights, MxNetEvaluator, SymbolBuilder, GLPFactorIter}
  
  val log = LoggerFactory.getLogger(getClass)
  // import scala.collection.JavaConversions._
  
  def getDeviceArray(appSettings: MxModelSettings) : Array[Context] = {
    val gpuContexts = appSettings.getGpus map {i => Context.gpu(i)}
    val cpuContexts = appSettings.getCpus map {i => Context.cpu(i)}
    (gpuContexts ++ cpuContexts).toArray
  }
  
  def getOptimizer(appSettings: MxModelSettings) = {
    val lr = appSettings.mxInitialLearnRate
    val rescale = appSettings.mxRescaleGrad
    appSettings.mxOptimizer match {      
      case "nag" => new NAG(learningRate = lr, momentum = appSettings.mxMomentum, wd = 0.0001f)
      case "adadelta" => new AdaDelta(rho = appSettings.mxRho, rescaleGradient = rescale)
      case "rmsprop" => new RMSProp(learningRate = lr, rescaleGradient = rescale)
      case "adam" => new Adam(learningRate = lr, clipGradient = appSettings.mxGradClip)
      case "adagrad" => new AdaGrad(learningRate = lr, rescaleGradient = rescale)
      case "sgld" => new SGLD(learningRate = lr, rescaleGradient = rescale, clipGradient = appSettings.mxGradClip)
      case _ => new SGD(learningRate = lr, momentum = appSettings.mxMomentum, wd = 0.0001f)
    }
  }
  
  def train(trVecs: Vector[GLPFactor], tstVecs: Vector[GLPFactor]) : Double = {
    log.info("Initiating training ...")
    val devices = getDeviceArray(appSettings)
    log.info("Getting symbol ")
    val sym     = (new SymbolBuilder).symbolFromSpec(appSettings.config)    
    val shape = Shape(nfs)
    val trIter = new GLPFactorIter(trVecs.toIterator, shape, appSettings.miniBatchSize)
    val tstIter = new GLPFactorIter(tstVecs.toIterator, shape, appSettings.miniBatchSize)
    val lr = appSettings.initialLearnRate
    val opt = getOptimizer(appSettings)
    val updater = new MxNetOptimizer(opt)
    val weights = new MxNetWeights(1.0f)
    log.info("Getting evaluator")
    val evaluator = new MxNetEvaluator(sym, devices, shape, appSettings.miniBatchSize, appSettings.modelFile, appSettings.saveFreq)
    val lg = evaluator.evaluateTrainingMiniBatch(trIter, tstIter, weights, updater, appSettings.numEpochs)
    lg.loss
  }
}

class LocalMxModelEvaluator(trData: Vector[GLPFactor], tstData: Vector[GLPFactor]) extends ModelEvaluator with Serializable {
  import org.slf4j.LoggerFactory
  
  val log = LoggerFactory.getLogger(getClass)

  override def evaluate(c: Seq[ModelConfig]): Seq[Double] = {
    val configs = c.toList
    val cvec = configs.par
    log.info(s"Evaluating model config sequence")
    cvec.tasksupport_=(new ForkJoinTaskSupport(new ForkJoinPool(cvec.length)))
    val accuracies = cvec map {config =>
      log.info(s"Getting learner")
      val learner = MxLearnerFactory.getLearnerInstance(config)
      log.info(s"Training model...")
      val acc = learner.train(trData, tstData)
      acc
    }
    log.info("Finished training all models locally..")
    accuracies.seq
  }
}
package org.mitre.mandolin.mselect

import org.mitre.mandolin.glp.{CategoricalGLPPredictor, ANNetwork, GLPFactor, GLPWeights, GLPComponentSet, MandolinMLPSettings,
  LType, TanHLType, ReluLType, InputLType, SparseInputLType, SoftMaxLType}
import org.mitre.mandolin.glp.local.{LocalGLPOptimizer, LocalProcessor}
import org.mitre.mandolin.predict.local.{LocalTrainTester, LocalEvalDecoder, NonExtractingEvalDecoder, LocalTrainer}
import org.mitre.mandolin.predict.DiscreteConfusion
import org.mitre.mandolin.util.LocalIOAssistant

import scala.collection.immutable.IndexedSeq
import scala.collection.mutable


trait LearnerInstance[T] extends LocalProcessor {
  def train(train: Vector[T], test: Option[Vector[T]]): Double
}


trait ModelSpaceBuilder {
  val reals = new mutable.MutableList[RealMetaParameter]
  val cats = new mutable.MutableList[CategoricalMetaParameter]
  val ints = new mutable.MutableList[IntegerMetaParameter]

  def withMetaParam(realMP: RealMetaParameter) = {
    reals += realMP
    this
  }

  def withMetaParam(catMP: CategoricalMetaParameter) = {
    cats += catMP
    this
  }
  def withMetaParam(intMP: IntegerMetaParameter) = {
    ints += intMP
    this
  }
  
  
}

class MandolinModelSpaceBuilder(ms: Option[ModelSpace]) extends ModelSpaceBuilder {
  
  def this(m: ModelSpace) = this(Some(m))
  def this() = this(None)
  
  val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  
  // initialize with modelspace
  ms foreach {ms => 
    ms.catMPs foreach withMetaParam
    ms.realMPs foreach withMetaParam
    ms.intMPs foreach withMetaParam
  }
       
  def build() : ModelSpace = build(0,0,false, None)  
  
  def checkModelSpace(s: MandolinMLPSettings) : Boolean = {
    ms match {
      case Some(ms) =>
        (ms.catMPs.forall {cmp =>
          val ok = s.config.hasPathOrNull(cmp.name)
          if (!ok) logger.error("Invalid categorical meta-parameter name: " + cmp.name)
          ok} && 
         ms.realMPs.forall {cmp => 
           val ok = s.config.hasPathOrNull(cmp.name)
           if (!ok) logger.error("Invalid real meta-parameter name: " + cmp.name)
           ok
           } &&
         ms.intMPs.forall {cmp => 
           val ok = s.config.hasPathOrNull(cmp.name)
           if (!ok) logger.error("Invalid integer meta-parameter name: " + cmp.name)
           ok
           } )
      case None => true
    }
  }
  
  def build(idim: Int, odim: Int, sparse: Boolean, appSettings: Option[MandolinMLPSettings]) : ModelSpace = {
    val appConfig = appSettings map {a => a.config.root.render()}
    val budget = appSettings match {case Some(m) => m.numEpochs case None => -1}
    if (appSettings.isDefined && !checkModelSpace(appSettings.get)) 
      throw new RuntimeException("Model appears to be defined over invalid configuration values")
    new ModelSpace(reals.toVector, cats.toVector, ints.toVector, idim, odim, appConfig, budget)    
  }
}

class MandolinModelInstance(appSettings: MandolinMLPSettings, config: ModelConfig, nn: ANNetwork) extends LearnerInstance[GLPFactor] {

  val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  
  def train(train: Vector[GLPFactor], test: Option[Vector[GLPFactor]]) : Double = {
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(appSettings, nn)
    val predictor = new CategoricalGLPPredictor(nn, true)
    val trainer = new LocalTrainer[GLPFactor, GLPFactor, GLPWeights](optimizer)
    
    test match {
      case Some(testD) =>
        logger.info("Model evaluation using fixed validation set...")
        val evPr = new NonExtractingEvalDecoder[GLPFactor,GLPWeights,Int,DiscreteConfusion](predictor)
        val (weights, trainLoss) = trainer.retrainWeights(train, appSettings.numEpochs)    
        val confusion = evPr.evalWithoutExtraction(test.getOrElse(train), weights)    
        val acc = confusion.getAccuracy
        acc
      case None =>
        logger.info("Model evaluation using cross validation ...")
        try {
          val xValidator = new LocalTrainTester(trainer,predictor,appSettings.numEpochs,1)
          logger.info("Got LocalTrainTester")
          val confusion = xValidator.crossValidate(train, 5)
          confusion.getAccuracy
        } catch {
          case e: Throwable =>
            e.printStackTrace()
            logger.info("Stack trace: " + e.toString())
            0.0
        }
                
    }
  }
  
}

object MandolinModelInstance {
  
  val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  
  def apply(config: ModelConfig) : MandolinModelInstance = {
    logger.info("Getting model evaluation instance")
    val cats: List[(String,Any)] = config.categoricalMetaParamSet.toList map {cm => (cm.getName,cm.getValue.s)}
    val reals : List[(String,Any)] = config.realMetaParamSet.toList map {cm => (cm.getName,cm.getValue.v)}
    val ints : List[(String,Any)] = config.intMetaParamSet.toList map {cm => (cm.getName, cm.getValue.v)}
    logger.info("Model space hyper-parameter ranges acquired from spec file ...")
    //val mspecValued = config.topoMPs map {ms => ms.getValue.v.s map {l => l.drawRandomValue.getValue} map {vl => getSpec(vl)}}
    val sets = config.serializedSettings match {case Some(s) => new MandolinMLPSettings(s) case None => new MandolinMLPSettings()}
    
    // val fullSpec : Vector[LType] = Vector(config.inLType) ++  Vector(config.outLType)
    // val net = ANNetwork(sets.ne, config.inDim, config.outDim)
    val fullSpec = org.mitre.mandolin.glp.ANNBuilder.getGLPSpec(sets.netspec, config.inDim, config.outDim)
    val net = ANNetwork(fullSpec, config.inDim, config.outDim)
    val allParams : Seq[(String,Any)] = (cats ++ reals ++ ints) toSeq   
    
    logger.info("Constructing new settings object with latest parameters")
    val settings = sets.withSets(allParams)
    logger.info("Built new settings object with latest parameters")
    new MandolinModelInstance(settings, config, net)
  }
}

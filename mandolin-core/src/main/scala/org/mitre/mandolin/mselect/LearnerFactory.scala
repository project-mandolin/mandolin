package org.mitre.mandolin.mselect

import org.mitre.mandolin.glp.{CategoricalGLPPredictor, ANNetwork, GLPFactor, GLPWeights, GLPComponentSet, GLPModelSettings,
  LType, TanHLType, ReluLType, InputLType, SparseInputLType, SoftMaxLType}
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
  val ints = new mutable.MutableList[IntegerMetaParameter]
  var topo : Option[TopologySpaceMetaParameter] = None

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
  
  def withMetaParam(t: TopologySpaceMetaParameter) = {
    topo = Some(t)
    this
  }

  
}

trait MandolinModelSpaceBuilder extends ModelSpaceBuilder {
  def build() : ModelSpace = build(0,0,false, None)  
  
  def build(idim: Int, odim: Int, sparse: Boolean, appSettings: Option[GLPModelSettings]) : ModelSpace = {    
    val it = if (sparse) LType(SparseInputLType, idim) else LType(InputLType, odim)
    // XXX - eventually pull out important parameters to preserve here and pass into model space
    /*
    val opts : Option[Seq[(String,Any)]] = appSettings map { a =>
      Seq(("mandolin.trainer.specification",a.netspec))
    } 
    * 
    */
    new ModelSpace(reals.toVector, cats.toVector, ints.toVector, topo, it, LType(SoftMaxLType, odim), idim, odim, None)    
  }
}


object MandolinModelFactory extends LearnerFactory[GLPFactor] {
  class MandolinGenericModelSpaceBuilder extends MandolinModelSpaceBuilder {
    
    def withRealMetaParams(rs: Vector[RealMetaParameter]) = rs foreach withMetaParam 
    def withCategoricalMetaParams(cats: Vector[CategoricalMetaParameter]) = cats foreach withMetaParam
    def withIntegerMetaParams(ints: Vector[IntegerMetaParameter]) = ints foreach withMetaParam
    def withTopologyMetaParam(topo: TopologySpaceMetaParameter) = withMetaParam(topo)  
  }
  
  override def getModelSpaceBuilder() : MandolinGenericModelSpaceBuilder = {
    new MandolinGenericModelSpaceBuilder
  }
  
  def getModelSpaceBuilder(ms: ModelSpace) : MandolinGenericModelSpaceBuilder = {
    val mm = new MandolinGenericModelSpaceBuilder
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
    
    //val mspecValued = config.topoMPs map {ms => ms.getValue.v.s map {l => l.drawRandomValue.getValue} map {vl => getSpec(vl)}}
    val hiddenLayers = config.topo.getOrElse(Vector())
    
    val fullSpec : Vector[LType] = Vector(config.inLType) ++  hiddenLayers ++ Vector(config.outLType)
    val net = ANNetwork(fullSpec, config.inDim, config.outDim)
    val allParams : Seq[(String,Any)] = (cats ++ reals ++ ints) toSeq 
    val completeParams = allParams ++ config.fixedSettingValues  // add in fixed settings
    val settings = (new GLPModelSettings()).withSets(completeParams)
    new MandolinModelInstance(settings, config, net)
  }
}

class MandolinModelInstance(appSettings: GLPModelSettings, config: ModelConfig, nn: ANNetwork) 
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

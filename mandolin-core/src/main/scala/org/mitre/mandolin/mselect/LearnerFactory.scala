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

  def build() : ModelSpace = build(0,0)  
  
  def build(idim: Int, odim: Int, sparse: Boolean = true, appSettings: Option[GLPModelSettings] = None) : ModelSpace = {    
    val it = if (sparse) LType(SparseInputLType, idim) else LType(InputLType, odim)
    new ModelSpace(reals.toVector, cats.toVector, ints.toVector, topo, it, LType(SoftMaxLType, odim), idim, odim, appSettings)    
  }
}


object GenericModelFactory extends LearnerFactory[GLPFactor] {
  class GenericModelSpaceBuilder extends ModelSpaceBuilder {
    
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
    ms.ms foreach {ms => mm.withTopologyMetaParam(ms) }
    mm
  }
  
  def getSpec(lsp: Tuple4Value[CategoricalValue, IntValue, RealValue, RealValue]) : LType = {
      val lt = lsp.v1.s match {case "TanHLType" => TanHLType case _ => ReluLType}
      val dim = lsp.v2.v
      val l1 = lsp.v3.v
      val l2 = lsp.v4.v
      LType(lt, dim, l1 = l1.toFloat, l2 = l2.toFloat)            
   }
  
  def getLearnerInstance(config: ModelConfig) : LearnerInstance[GLPFactor] = {
    val cats: List[(String,Any)] = config.categoricalMetaParamSet.toList map {cm => (cm.getName,cm.getValue.s)}
    val reals : List[(String,Any)] = config.realMetaParamSet.toList map {cm => (cm.getName,cm.getValue.v)}
    val ints : List[(String,Any)] = config.intMetaParamSet.toList map {cm => (cm.getName, cm.getValue.v)}
    
    val mspecValued = config.ms map {ms => ms.getValue.v.s map {l => l.drawRandomValue.getValue} map {vl => getSpec(vl)}}
    val hiddenLayers = mspecValued.getOrElse(Vector())
    
    val fullSpec : Vector[LType] = Vector(config.inLType) ++  hiddenLayers ++ Vector(config.outLType)
    val net = ANNetwork(fullSpec, config.inDim, config.outDim) // val net = ANNetwork(fullSpec, config.inDim, config.outDim)
    val allParams : Seq[(String,Any)] = (cats ++ reals ++ ints) toSeq 
    val settings = config.optionalSettings.getOrElse(new GLPModelSettings()).withSets(allParams)
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

//trait MandolinLogisticRegressionFactory extends LearnerFactory[GLPFactor]
/*
object MandolinLogisticRegressionFactory extends LearnerFactory[GLPFactor] {
  

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
    
    def defineModelTopology(n: String, lowBound: Int, upBound: Int) = {
      withMetaParam(new LayerMetaParameter(
          n,
          new TupleSet4(
          new CategoricalMetaParameter("ltype", new CategoricalSet(Vector("TanHLType"))),
          new IntegerMetaParameter("dim", new IntSet(lowBound, upBound)),
          new RealMetaParameter("l1", new RealSet(0.0, 0.01)),
          new RealMetaParameter("l2", new RealSet(0.0, 0.01)))))
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
    
    def getSpec(vs: ValuedMetaParameter[Tuple4Value[CategoricalValue, IntValue, RealValue, RealValue]]) : LType = {
      val lsp = vs.getValue
      val lt = lsp.v1.s match {case "TanHLType" => TanHLType case _ => ReluLType}
      val dim = lsp.v2.v
      val l1 = lsp.v3.v
      val l2 = lsp.v4.v
      LType(lt, dim, l1 = l1.toFloat, l2 = l2.toFloat)            
      }

    val mspecValued = config.ms map {m => m.drawRandomValue} map getSpec
    // this currently hard-codes the input to SparseInputLType and output to SoftMaxLType
    val fullSpec : Vector[LType] = Vector(LType(SparseInputLType)) ++  mspecValued ++ Vector(LType(SoftMaxLType))
    val net = ANNetwork(fullSpec, config.inDim, config.outDim)
    val allParams : Seq[(String,Any)] = (cats ++ reals) toSeq 
    val settings = new GLPModelSettings().withSets(allParams)
    new MandolinModelInstance(settings, config, net)
  }


}
*/
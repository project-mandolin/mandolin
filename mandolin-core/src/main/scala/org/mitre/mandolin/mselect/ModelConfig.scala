package org.mitre.mandolin.mselect

import org.mitre.mandolin.glp.{GLPModelSettings,ANNetwork, TanHLType, ReluLType, LType, InputLType, SparseInputLType, SoftMaxLType}


abstract class AbstractModelConfig(
    val realMetaParamSet: Vector[ValuedMetaParameter[RealValue]],
                   val categoricalMetaParamSet: Vector[ValuedMetaParameter[CategoricalValue]],
                   val intMetaParamSet: Vector[ValuedMetaParameter[IntValue]],
                   val inDim: Int,
                   val outDim: Int,
                   val serializedSettings : Option[String]
    ) extends Serializable
/**
  * A configuration is a set of MetaParameters set to particular values.
  * A GLP model setting can be passed in so that all the static (unchanging) learning
  * settings are provided as part of the model config.
  */
class ModelConfig(
                   _realMetaParamSet: Vector[ValuedMetaParameter[RealValue]],
                   _categoricalMetaParamSet: Vector[ValuedMetaParameter[CategoricalValue]],
                   _intMetaParamSet: Vector[ValuedMetaParameter[IntValue]],                   
                   val topo : Option[Vector[LType]],
                   val inLType : LType,
                   val outLType: LType,
                   _inDim: Int,
                   _outDim: Int,
                   _serializedSettings : Option[String]) 
                   extends AbstractModelConfig(_realMetaParamSet, _categoricalMetaParamSet, _intMetaParamSet, _inDim, _outDim, _serializedSettings) 
with Serializable {

  override def toString(): String = {
      val reals = realMetaParamSet.map { mp =>
      mp.getName + ":" + mp.getValue.v
    }.mkString(" ")
    val cats = categoricalMetaParamSet.map { mp => mp.getName + "_" + mp.getValue.s }.mkString(" ")
    val ints = intMetaParamSet map { mp =>
      mp.getName + ":" + mp.getValue.v
      } mkString(" ")
    val layerInfo = topo 
    val numHiddenLayers = layerInfo.getOrElse(Vector()).size
    var totalWeights = 0
    layerInfo match {
        case Some(ls) =>
          for (i <- 0 until ls.length) {
            val n = if (i == 0) inDim * ls(i).dim else ls(i).dim * ls(i-1).dim 
            totalWeights += n
          }
          totalWeights += ls(ls.length - 1).dim * outDim // add output weights
        case None => totalWeights = inDim * outDim
      }
    reals + " " + ints + " " + cats 
  }
}

abstract class AbstractModelSpace(
    val realMPs: Vector[RealMetaParameter], 
    val catMPs: Vector[CategoricalMetaParameter],
    val intMPs: Vector[IntegerMetaParameter],
    val idim: Int,
    val odim: Int,
    val settings: Option[String]) {
  def drawRandom : ModelConfig
}

/**
 * Defines a space of model configurations as sets of MetaParameter objects - real,
 * categorical or complex. It also includes optional hard-coded input and output dimensions,
 * which can be gleaned from the data automatically rather than specified by the user.
 * @author wellner@mitre.org
 */
class ModelSpace(_realMPs: Vector[RealMetaParameter], _catMPs: Vector[CategoricalMetaParameter],
    _intMPs: Vector[IntegerMetaParameter],
    val topoMPs: Option[TopologySpaceMetaParameter],
    val inLType: LType,
    val outLType: LType,
    _idim: Int,
    _odim: Int,
    _settings: Option[String]) 
    extends AbstractModelSpace(_realMPs, _catMPs, _intMPs, _idim, _odim, _settings) with Serializable {
  
  def getSpec(lsp: Tuple4Value[CategoricalValue, IntValue, RealValue, RealValue]) : LType = {
      val lt = lsp.v1.s match {case "TanHLType" => TanHLType case _ => ReluLType}
      val dim = lsp.v2.v
      val l1 = lsp.v3.v
      val l2 = lsp.v4.v
      LType(lt, dim, l1 = l1.toFloat, l2 = l2.toFloat)            
   }    
    
  def this(rmps: Vector[RealMetaParameter], cmps: Vector[CategoricalMetaParameter], ints: Vector[IntegerMetaParameter]) =
    this(rmps, cmps, ints, None, LType(InputLType), LType(SoftMaxLType), 0,0, None)

  def drawRandom: ModelConfig = {
    val realValued = realMPs map { mp => mp.drawRandomValue }
    val catValued = catMPs map { mp => mp.drawRandomValue }
    val intValued = intMPs map {mp => mp.drawRandomValue }
    if (topoMPs.isDefined) {
      val topology = topoMPs.get.drawRandomValue
      val mspecValued = topology.getValue.v.s map {l => l.drawRandomValue.getValue} map {vl => getSpec(vl)}
      new ModelConfig(realValued, catValued, intValued, Some(mspecValued), inLType, outLType, idim, odim, settings)
    } else {
      new ModelConfig(realValued, catValued, intValued, None, inLType, outLType, idim, odim, settings)
    }
  }
}





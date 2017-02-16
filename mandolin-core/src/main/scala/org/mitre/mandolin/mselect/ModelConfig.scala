package org.mitre.mandolin.mselect

import org.mitre.mandolin.glp.{ANNetwork, TanHLType, ReluLType, LType, InputLType, SparseInputLType, SoftMaxLType}


/**
  * A configuration is a set of MetaParameters set to particular values.
  */
class ModelConfig(
                   val realMetaParamSet: Vector[ValuedMetaParameter[RealValue]],
                   val categoricalMetaParamSet: Vector[ValuedMetaParameter[CategoricalValue]],
                   val intMetaParamSet: Vector[ValuedMetaParameter[IntValue]],
                   val ms : Option[ValuedMetaParameter[ListValue[SetValue[LayerMetaParameter]]]],
                   val inLType : LType,
                   val outLType: LType,
                   val inDim: Int,
                   val outDim: Int) extends Serializable {
                   // val inSpec : ValuedMetaParameter[Tuple2Value[CategoricalValue,RealValue]],
                   //val hiddenSpec : Vector[ValuedMetaParameter[Tuple4Value[CategoricalValue,IntValue,RealValue,RealValue]]],
                   //val outSpec : ValuedMetaParameter[Tuple3Value[CategoricalValue,RealValue,RealValue]],
                   
                   //
                   //val nn: ANNetwork) extends Serializable {

  override def toString(): String = {
      val reals = realMetaParamSet.map { mp =>
      mp.getName + ":" + mp.getValue.v
    }.mkString(" ")
    val cats = categoricalMetaParamSet.map { mp => mp.getName + ":" + mp.getValue.s }.mkString(" ")
    val ints = intMetaParamSet map { mp =>
      mp.getName + ":" + mp.getValue.v
      } mkString(" ")
    val layerInfo = ms map {lm => lm.getValue.v.s}
    val numHiddenLayers = layerInfo.getOrElse(Vector()).size 
    reals + " " + ints + " " + cats + " numHiddenLayers:" + numHiddenLayers
  }
}

/**
 * Defines a space of model configurations as sets of MetaParameter objects - real,
 * categorical or complex. It also includes optional hard-coded input and output dimensions,
 * which can be gleaned from the data automatically rather than specified by the user.
 * @author wellner@mitre.org
 */
class ModelSpace(val realMPs: Vector[RealMetaParameter], val catMPs: Vector[CategoricalMetaParameter],
    val intMPs: Vector[IntegerMetaParameter],
    val ms: Option[TopologySpaceMetaParameter],
    val inLType: LType,
    val outLType: LType,
    val idim: Int,
    val odim: Int) {
    
  def this(rmps: Vector[RealMetaParameter], cmps: Vector[CategoricalMetaParameter], ints: Vector[IntegerMetaParameter]) =
    this(rmps, cmps, ints, None, LType(InputLType), LType(SoftMaxLType), 0,0)

  def drawRandom: ModelConfig = {
    val realValued = realMPs map { mp => mp.drawRandomValue }
    val catValued = catMPs map { mp => mp.drawRandomValue }
    val intValued = intMPs map {mp => mp.drawRandomValue }
    if (ms.isDefined) {
      val topology = ms.get.drawRandomValue
      new ModelConfig(realValued, catValued, intValued, Some(topology), inLType, outLType, idim, odim)
    } else {
      new ModelConfig(realValued, catValued, intValued, None, inLType, outLType, idim, odim)
    }
  }
}





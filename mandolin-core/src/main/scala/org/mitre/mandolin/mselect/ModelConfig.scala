package org.mitre.mandolin.mselect

import org.mitre.mandolin.glp.{ANNetwork, TanHLType, ReluLType, LType, InputLType, SparseInputLType, SoftMaxLType}


/**
  * A configuration is a set of MetaParameters set to particular values.
  */
class ModelConfig(
                   val realMetaParamSet: Vector[ValuedMetaParameter[RealValue]],
                   val categoricalMetaParamSet: Vector[ValuedMetaParameter[CategoricalValue]],
                   val ms : Vector[LayerMetaParameter],
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
    reals + " " + cats
  }
}

/**
 * Defines a space of model configurations as sets of MetaParameter objects - real,
 * categorical or complex. It also includes optional hard-coded input and output dimensions,
 * which can be gleaned from the data automatically rather than specified by the user.
 * @author wellner@mitre.org
 */
class ModelSpace(val realMPs: Vector[RealMetaParameter], val catMPs: Vector[CategoricalMetaParameter], 
    val ms: Vector[LayerMetaParameter],
    val inputDim: Int = 0,
    val outputDim: Int = 0) {
  
  
  def this(rmps: Vector[RealMetaParameter], cmps: Vector[CategoricalMetaParameter]) =
    this(rmps, cmps, Vector())

  def drawRandom: ModelConfig = {
    val realValued = realMPs map { mp => mp.drawRandomValue }
    val catValued = catMPs map { mp => mp.drawRandomValue }    
    new ModelConfig(realValued, catValued, ms, inputDim, outputDim)
  }
}





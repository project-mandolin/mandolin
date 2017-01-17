package org.mitre.mandolin.mselect

import org.mitre.mandolin.glp.ANNetwork
/**
 * A configuration is a set of MetaParameters set to particular values.
 */
class ModelConfig(
    val realMetaParamSet: Vector[ValuedMetaParameter[RealValue]], 
    val categoricalMetaParamSet: Vector[ValuedMetaParameter[CategoricalValue]],
    val mSpec: ANNetwork)  extends Serializable {
}

class ModelSpace(realMPs: Vector[RealMetaParameter], catMPs: Vector[CategoricalMetaParameter], nn: ANNetwork) {
  
  def this(rmps: Vector[RealMetaParameter], cmps: Vector[CategoricalMetaParameter]) = 
    this(rmps, cmps, ANNetwork(IndexedSeq()))
    
  def drawRandom : ModelConfig = {
    val realValued = realMPs map {mp => mp.drawRandomValue}
    val catValued = catMPs map {mp => mp.drawRandomValue}
    new ModelConfig(realValued, catValued, nn)
  }
}




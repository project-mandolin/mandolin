package org.mitre.mandolin.mselect

/**
 * A configuration is a set of MetaParameters set to particular values.
 */
class ModelConfig(
    val realMetaParamSet: Vector[ValuedMetaParameter[RealValue]], 
    val categoricalMetaParamSet: Vector[ValuedMetaParameter[CategoricalValue]]) {

}

class ModelSpace(realMPs: Vector[RealMetaParameter], catMPs: Vector[CategoricalMetaParameter]) {
  
  def drawRandom : ModelConfig = {
    val realValued = realMPs map {mp => mp.drawRandomValue}
    val catValued = catMPs map {mp => mp.drawRandomValue}
    new ModelConfig(realValued, catValued)
  }
}



object ModelConfig {
  def apply() = new ModelConfig(Vector(),Vector())
}
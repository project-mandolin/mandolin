package org.mitre.mandolin.mselect

import org.mitre.mandolin.glp.{GLPModelSettings,ANNetwork, TanHLType, ReluLType, LType, InputLType, SparseInputLType, SoftMaxLType}


/**
  * A configuration is a set of MetaParameters set to particular values.
  * A GLP model setting can be passed in so that all the static (unchanging) learning
  * settings are provided as part of the model config.
  */
class ModelConfig(
                   val realMetaParamSet: Vector[ValuedMetaParameter[RealValue]],
                   val categoricalMetaParamSet: Vector[ValuedMetaParameter[CategoricalValue]],
                   val intMetaParamSet: Vector[ValuedMetaParameter[IntValue]],
                   val topo : Option[Vector[LType]],
                   val inLType : LType,
                   val outLType: LType,
                   val inDim: Int,
                   val outDim: Int,
                   val optionalSettings: Option[GLPModelSettings] = None) extends Serializable {
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
    reals + " " + ints + " " + cats + " numHiddenLayers:" + numHiddenLayers + " totalWeights:" + totalWeights
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
    val topoMPs: Option[TopologySpaceMetaParameter],
    val inLType: LType,
    val outLType: LType,
    val idim: Int,
    val odim: Int,
    val settings: Option[GLPModelSettings] = None) {
  
  def getSpec(lsp: Tuple4Value[CategoricalValue, IntValue, RealValue, RealValue]) : LType = {
      val lt = lsp.v1.s match {case "TanHLType" => TanHLType case _ => ReluLType}
      val dim = lsp.v2.v
      val l1 = lsp.v3.v
      val l2 = lsp.v4.v
      LType(lt, dim, l1 = l1.toFloat, l2 = l2.toFloat)            
   }    
    
  def this(rmps: Vector[RealMetaParameter], cmps: Vector[CategoricalMetaParameter], ints: Vector[IntegerMetaParameter]) =
    this(rmps, cmps, ints, None, LType(InputLType), LType(SoftMaxLType), 0,0)

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





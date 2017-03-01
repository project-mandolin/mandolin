package org.mitre.mandolin.mx

import org.mitre.mandolin.optimize.Weights
import ml.dmlc.mxnet.NDArray

class MxNetWeights(var argParams: Option[Map[String, NDArray]], var auxParams: Option[Map[String, NDArray]], m: Float) 
extends Weights[MxNetWeights](m) with Serializable {
  
  def this(ap: Map[String,NDArray], ax: Map[String, NDArray], m: Float) = this(Some(ap), Some(ax), m)
  def this(m: Float) = this(None, None, m)

  val numWeights = -1
  def compress(): Unit = {}
  def decompress(): Unit = {}
  def weightAt(i: Int) = throw new RuntimeException("Not implemented")
  
  def setArgParams(m: Map[String, NDArray]) = {
    argParams foreach {mp => mp.foreach{case (_,v) => v.dispose()}} // explicitly deallocate old params
    argParams = Some(m)
  }
  def setAuxParams(m: Map[String, NDArray]) = {
    auxParams foreach {mp => mp.foreach{case (_,v) => v.dispose()}} // explicitly deallocate old params
    auxParams = Some(m)
  }
  def getArgParams = argParams.get
  def getAuxParams = auxParams.get
  
  def compose(otherWeights: MxNetWeights) : MxNetWeights = {
    this *= mass
    otherWeights *= otherWeights.mass
    this += otherWeights
    val nmass = mass + otherWeights.mass
    this *= (1.0f / nmass)
    new MxNetWeights(this.argParams, this.auxParams, nmass)
  }
  
  def timesEquals(v: Float) = argParams foreach {ap =>
      ap foreach {case (s,arr) => arr *= v }
      ap foreach {case (s,arr) => arr *= v }   
  }
  
  def addEquals(other: MxNetWeights) = argParams foreach {ap => 
    ap foreach {case (s,arr) => arr += (other.argParams.get(s))}
    ap foreach {case (s,arr) => arr += (other.auxParams.get(s))}
  }
  
  def add(other: MxNetWeights) = {
    addEquals(other)
    this
  }
  
  def l2norm = throw new RuntimeException("Norm not implemented yet")
  def updateFromArray(ar: Array[Float]) = throw new RuntimeException("updateFromArray not implemented yet")
  def updateFromArray(ar: Array[Double]) = throw new RuntimeException("updateFromArray not implemented yet")
  
  def copy() = throw new RuntimeException("Copy should/can not be performed on MxNetWeights")
  
  def asArray() = throw new RuntimeException("asArray not implemented")
  def asTensor1() = throw new RuntimeException("asTensor1 not implemented")
  
  

}
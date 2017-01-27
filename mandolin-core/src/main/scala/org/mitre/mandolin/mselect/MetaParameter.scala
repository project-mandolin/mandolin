package org.mitre.mandolin.mselect

abstract class ValueSet[T <: MPValue] extends Serializable {
  
}

class CategoricalSet(valueSet: Vector[String]) extends ValueSet[CategoricalValue] with Serializable {
  def size = valueSet.size
  def apply(i: Int) = valueSet(i)
}

class RealSet(val lower: Double, val upper: Double) extends ValueSet[RealValue] with Serializable {
  def getDiff = upper - lower
  
}

abstract class MPValue extends Serializable
case class CategoricalValue(s: String) extends MPValue with Serializable
case class RealValue(v: Double) extends MPValue with Serializable

/**
 * Objects of this class represent meta parameters for learning algorithms.
 * These include hyper-parameters but also parameters that adjust architecture,
 * specify prior distributions, etc.
 */
abstract class MetaParameter[T <: MPValue](val name: String, valueSet: ValueSet[T]) extends Serializable {
  def drawRandomValue : ValuedMetaParameter[T]
}

class RealMetaParameter(n: String, vs: RealSet) extends MetaParameter[RealValue](n, vs) with Serializable {

  def drawRandomValue : ValuedMetaParameter[RealValue] = {
    new ValuedMetaParameter(RealValue(util.Random.nextDouble() * vs.getDiff + vs.lower), this)
  }
}

class CategoricalMetaParameter(n: String, val valSet: CategoricalSet) extends MetaParameter[CategoricalValue](n, valSet) with Serializable {

  def drawRandomValue : ValuedMetaParameter[CategoricalValue] = {
    new ValuedMetaParameter(CategoricalValue(valSet(util.Random.nextInt(valSet.size))), this)
  }
}

/**
 * A Meta Parameter with a particular instantiated value
 */
class ValuedMetaParameter[T <: MPValue](v: T, mp: MetaParameter[T]) extends Serializable {

  def getValue = v
  def getName = mp.name

}



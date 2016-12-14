package org.mitre.mandolin.mselect

abstract class ValueSet[T <: MPValue] {
}

class CategoricalSet(valueSet: Vector[String]) extends ValueSet[CategoricalValue] {
  def size = valueSet.size
  def apply(i: Int) = valueSet(i)
}

class RealSet(val lower: Double, val upper: Double) extends ValueSet[RealValue] {
  def getDiff = upper - lower
  
}

abstract class MPValue
case class CategoricalValue(s: String) extends MPValue
case class RealValue(v: Double) extends MPValue

/**
 * Objects of this class represent meta parameters for learning algorithms.
 * These include hyper-parameters but also parameters that adjust architecture,
 * specify prior distributions, etc.
 */
abstract class MetaParameter[T <: MPValue](val name: String, val valueSet: ValueSet[T])  {
  def drawRandomValue : ValuedMetaParameter[T]
}

class RealMetaParameter(n: String, vs: RealSet) extends MetaParameter[RealValue](n, vs) {

  def drawRandomValue : ValuedMetaParameter[RealValue] = {
    new ValuedMetaParameter(RealValue(util.Random.nextDouble() * vs.getDiff + vs.lower), this)
  }
}

class CategoricalMetaParameter(n: String, vs: CategoricalSet) extends MetaParameter[CategoricalValue](n, vs) {

  def drawRandomValue : ValuedMetaParameter[CategoricalValue] = {
    new ValuedMetaParameter(CategoricalValue(vs(util.Random.nextInt(vs.size))), this)
  }
}

/**
 * A Meta Parameter with a particular instantiated value
 */
class ValuedMetaParameter[T <: MPValue](v: T, mp: MetaParameter[T]) {

  def getValue = v
  def getName = mp.name

}



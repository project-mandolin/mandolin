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

class IntSet(val lower: Int, val upper: Int) extends ValueSet[IntValue] with Serializable {
  def getDiff = upper - lower
}

// these can represent complex tuple-valued meta-parameters
class TupleSet2[T1 <: MPValue, T2 <: MPValue](val e1: MetaParameter[T1], val e2: MetaParameter[T2]) extends ValueSet[Tuple2Value[T1,T2]] with Serializable
class TupleSet3[T1 <: MPValue, T2 <: MPValue, T3 <: MPValue](val e1: MetaParameter[T1], val e2: MetaParameter[T2], val e3: MetaParameter[T3]) extends ValueSet[Tuple3Value[T1,T2,T3]] with Serializable
class TupleSet4[T1 <: MPValue, T2 <: MPValue, T3 <: MPValue, T4 <: MPValue](val e1: MetaParameter[T1], val e2: MetaParameter[T2], val e3: MetaParameter[T3], val e4: MetaParameter[T4]) 
extends ValueSet[Tuple4Value[T1,T2,T3,T4]] with Serializable
// class TupleSet3[T1 <: MPValue,T2 <: MPValue,T3 <: MPValue](val tuple: (ValueSet[T1],ValueSet[T2],ValueSet[T3])) extends Serializable
// class TupleSet4[T1 <: MPValue,T2 <: MPValue,T3 <: MPValue,T4 <: MPValue](val tuple: (ValueSet[T1],ValueSet[T2],ValueSet[T3],ValueSet[T4])) extends Serializable

abstract class MPValue extends Serializable
case class CategoricalValue(s: String) extends MPValue with Serializable
case class RealValue(v: Double) extends MPValue with Serializable
case class IntValue(v: Int) extends MPValue with Serializable
case class Tuple2Value[T1,T2](v1:T1, v2: T2) extends MPValue with Serializable
case class Tuple3Value[T1,T2,T3](v1:T1, v2: T2, v3: T3) extends MPValue with Serializable
case class Tuple4Value[T1,T2,T3,T4](v1:T1, v2: T2, v3: T3, v4: T4) extends MPValue with Serializable

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

class Tuple2MetaParameter[T1 <: MPValue, T2 <: MPValue](n: String, ts: TupleSet2[T1,T2]) extends MetaParameter[Tuple2Value[T1,T2]](n,ts) with Serializable {
  
  def drawRandomValue : ValuedMetaParameter[Tuple2Value[T1,T2]] = {
    val first = ts.e1
    val second = ts.e2
    val v1 = first.drawRandomValue.getValue
    val v2 = second.drawRandomValue.getValue
    new ValuedMetaParameter(Tuple2Value[T1,T2](v1,v2), this)
  }
}

class Tuple3MetaParameter[T1 <: MPValue, T2 <: MPValue, T3 <: MPValue](n: String, ts: TupleSet3[T1,T2,T3]) extends MetaParameter[Tuple3Value[T1,T2,T3]](n,ts) with Serializable {
  
  def drawRandomValue : ValuedMetaParameter[Tuple3Value[T1,T2,T3]] = {
    val v1 = ts.e1.drawRandomValue.getValue
    val v2 = ts.e2.drawRandomValue.getValue
    val v3 = ts.e3.drawRandomValue.getValue
    new ValuedMetaParameter(Tuple3Value[T1,T2,T3](v1,v2,v3), this)
  }
}

class Tuple4MetaParameter[T1 <: MPValue, T2 <: MPValue, T3 <: MPValue, T4 <: MPValue](n: String, ts: TupleSet4[T1,T2,T3,T4]) 
extends MetaParameter[Tuple4Value[T1,T2,T3,T4]](n,ts) with Serializable {
  
  def drawRandomValue : ValuedMetaParameter[Tuple4Value[T1,T2,T3,T4]] = {
    val v1 = ts.e1.drawRandomValue.getValue
    val v2 = ts.e2.drawRandomValue.getValue
    val v3 = ts.e3.drawRandomValue.getValue
    val v4 = ts.e4.drawRandomValue.getValue
    new ValuedMetaParameter(Tuple4Value[T1,T2,T3,T4](v1,v2,v3,v4), this)
  }
}

/**
 * A Meta Parameter with a particular instantiated value
 */
class ValuedMetaParameter[T <: MPValue](v: T, mp: MetaParameter[T]) extends Serializable {

  def getValue = v
  def getName = mp.name

}



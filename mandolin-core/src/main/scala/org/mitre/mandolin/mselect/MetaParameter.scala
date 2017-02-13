package org.mitre.mandolin.mselect

sealed abstract class ValueSet[T <: MPValue] extends Serializable 

case class CategoricalSet(valueSet: Vector[String]) extends ValueSet[CategoricalValue] with Serializable {
  def size = valueSet.size
  def apply(i: Int) = valueSet(i)
}

case class RealSet(val lower: Double, val upper: Double) extends ValueSet[RealValue] with Serializable {
  def getDiff = upper - lower  
}

case class IntSet(val lower: Int, val upper: Int) extends ValueSet[IntValue] with Serializable {
  def getDiff = upper - lower
}



// these can represent complex tuple-valued meta-parameters
case class TupleSet2[T1 <: MPValue, T2 <: MPValue](val e1: MetaParameter[T1], val e2: MetaParameter[T2]) extends ValueSet[Tuple2Value[T1,T2]] with Serializable
case class TupleSet3[T1 <: MPValue, T2 <: MPValue, T3 <: MPValue](val e1: MetaParameter[T1], val e2: MetaParameter[T2], val e3: MetaParameter[T3]) extends ValueSet[Tuple3Value[T1,T2,T3]] with Serializable
case class TupleSet4[T1 <: MPValue, T2 <: MPValue, T3 <: MPValue, T4 <: MPValue](val e1: MetaParameter[T1], val e2: MetaParameter[T2], val e3: MetaParameter[T3], val e4: MetaParameter[T4]) 
extends ValueSet[Tuple4Value[T1,T2,T3,T4]] with Serializable

case class ListSet[T <: MPValue](li: Vector[MetaParameter[T]]) extends ValueSet[ListValue[T]] with Serializable {
  def size = li.size
  def apply(i: Int) = li(i)
} 


trait MPValue extends Serializable
case class CategoricalValue(s: String) extends MPValue with Serializable
case class RealValue(v: Double) extends MPValue with Serializable
case class IntValue(v: Int) extends MPValue with Serializable
case class Tuple2Value[T1,T2](v1:T1, v2: T2) extends MPValue with Serializable
case class Tuple3Value[T1,T2,T3](v1:T1, v2: T2, v3: T3) extends MPValue with Serializable
case class Tuple4Value[T1,T2,T3,T4](v1:T1, v2: T2, v3: T3, v4: T4) extends MPValue with Serializable
case class ListValue[T](v: T) extends MPValue with Serializable
// types a bit tricky below - need SetValue to be both an MPValue and a ValueSet
case class SetValue[T](s: Vector[T]) extends ValueSet[SetValue[T]] with MPValue with Serializable


/**
 * Objects of this class represent meta parameters for learning algorithms.
 * These include hyper-parameters but also parameters that adjust architecture,
 * specify prior distributions, etc.
 */
sealed abstract class MetaParameter[T <: MPValue](val name: String, valueSet: ValueSet[T]) extends Serializable {
  def drawRandomValue : ValuedMetaParameter[T]
}

case class RealMetaParameter(n: String, val vs: RealSet) extends MetaParameter[RealValue](n, vs) with Serializable {

  def drawRandomValue : ValuedMetaParameter[RealValue] = {
    new ValuedMetaParameter(RealValue(util.Random.nextDouble() * vs.getDiff + vs.lower), this)
  }
}

// handles arbitrary lists of meta parameters as another meta-parameter
// draw random just selects one

// this is just a wrapper so that we can have vectors of meta-parameters treated as single meta-parameters
case class FixedSetMetaParameter[T <: SetValue[T]](n: String, fixedSet: SetValue[T]) extends MetaParameter[SetValue[T]](n, fixedSet) with Serializable {
  def drawRandomValue : ValuedMetaParameter[SetValue[T]] = new ValuedMetaParameter(fixedSet, this)
}


case class CategoricalMetaParameter(n: String, val valSet: CategoricalSet) extends MetaParameter[CategoricalValue](n, valSet) with Serializable {

  def drawRandomValue : ValuedMetaParameter[CategoricalValue] = {
    new ValuedMetaParameter(CategoricalValue(valSet(util.Random.nextInt(valSet.size))), this)
  }
}

case class IntegerMetaParameter(n: String, val vs: IntSet) extends MetaParameter[IntValue](n, vs) with Serializable {
  def drawRandomValue : ValuedMetaParameter[IntValue] = {
    new ValuedMetaParameter(IntValue(util.Random.nextInt(vs.upper - vs.lower) + vs.lower), this)
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

case class ListMetaParameter[T <: MPValue](n: String, liSet: ListSet[T]) extends MetaParameter[ListValue[T]](n, liSet) with Serializable {
  def drawRandomValue : ValuedMetaParameter[ListValue[T]] = {
    val mp = liSet(util.Random.nextInt(liSet.size))
    val ivalue = mp.drawRandomValue
    new ValuedMetaParameter(ListValue(ivalue.getValue), this)
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

class LayerMetaParameter(_n: String, _ts: TupleSet4[CategoricalValue, IntValue, RealValue, RealValue]) 
extends Tuple4MetaParameter[CategoricalValue, IntValue, RealValue, RealValue](_n, _ts)

// MPValue is a trait - and so we can add this to any metaparameter that also needs to serve as a value type
class TopologyMetaParameter(_n: String, _layers: Vector[LayerMetaParameter])
extends MetaParameter[SetValue[LayerMetaParameter]](_n, SetValue(_layers)) with MPValue {
  def drawRandomValue : ValuedMetaParameter[SetValue[LayerMetaParameter]] = 
    new ValuedMetaParameter(SetValue(_layers), this)
}

class TopologySpaceMetaParameter(_n: String, _topos: ListSet[SetValue[LayerMetaParameter]])
extends ListMetaParameter[SetValue[LayerMetaParameter]](_n, _topos) {
  def this(nn: String) = this(nn, ListSet(Vector()))
  
}

/**
 * A Meta Parameter with a particular instantiated value
 */
class ValuedMetaParameter[T <: MPValue](v: T, mp: MetaParameter[T]) extends Serializable {

  def getValue = v
  def getName = mp.name

}



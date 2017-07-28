package org.mitre.mandolin.util
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import cern.colt.map.OpenIntDoubleHashMap
import cern.colt.function.IntDoubleProcedure

/**
 * Abstract top-level class for all tensor objects - sparse or dense.
 * Note that entries are always assumed to be Float types.
 * The class member densityRatio indicates the ratio at which sparse
 * representations are converted to dense (and possibly vice-versa).
 * A value of 1.0 indicates that sparse tensors will never be converted to 
 * dense representations.
 * 
 * @author Ben Wellner
 * @param densityRatio The ratio of non-zero entries to the size of the Tensor at 
 * which the `Tensor` will automatically switch to a dense representation.
 */
abstract class Tensor(val densityRatio: Float) extends Serializable {

  protected val size: Int
  def getSize: Int

  /** IndexedSeq with the size of each dimension */
  def getDimSizes: collection.immutable.IndexedSeq[Int]
}

/**
 *  Represents a 1-dimensional vector
 */
abstract class Tensor1(dr: Float = 1.0f) extends Tensor(dr) {

  def getDimSizes = collection.immutable.IndexedSeq(getDim)
  
  /** Assign values in tensor `v` to this `Tensor1` */
  def :=(v: Tensor1) : Unit
  
  /** Assign value `v` to each component in `Tensor1` */
  def :=(v: Float) : Unit
  
  /** Gets the number of dimensions for this `Tensor1` - i.e. vector */
  def getDim: Int

  /** Access the ''i''th element of the `Tensor1` object */
  def apply(i: Int): Float

  /** Update the ''i''th element of the `Tensor1` object */
  def update(i: Int, v: Float): Unit

  def mapInPlace(fn: ((Int,Float) => Float)): Unit
  
  /** Dot product */
  def *(t: Tensor1) : Float
  
  /** Matrix-scalar product. Returns new `Tensor1` */
  def *(v: Float) : Tensor1

  /** Add the value `v` to the ''i''th element of the Tensor */
  def +=(i: Int, v: Float): Unit

  /** Add the value `v` to the ''i''th element of the Tensor */
  def -=(i: Int, v: Float): Unit

  /** Adds the value `v` to all */
  def +=(v: Float): Unit

  /** Multiply each component of the Tensor by `v` */
  def *=(v: Float): Unit

  /** Divide each component of the Tensor by `v` */
  def /=(v: Float): Unit = *=(1.0f / v)

  /** Component-wise addition */
  def +=(t: Tensor1): Unit

  /** Component-wise addition */
  def -=(t: Tensor1): Unit
  
  /** Raise each element of this tensor to a power */
  def ^=(v: Float): Unit

  /**
   * Multiplies each component of '''this''' Tensor by `v` and adds it to the corresponding component
   *  in the provided array `ar`
   */
  def *+=(v: Float, ar: Array[Float]): Unit

  /**
   * Takes each component of '''this''' Tensor and adds it to the corresponding component
   *  in the provided array `ar`
   */
  def +=(ar: Array[Float]): Unit = *+=(1.0f, ar)

  /**
   * Multiplies each component of '''this''' Tensor by `v` and subtracts it from the corresponding component
   *  in the provided array `ar`
   */
  def *-=(v: Float, ar: Array[Float]): Unit

  /**
   * Multiplies each component of '''this''' Tensor by the corresponding component in `a1`
   *  and adds the result to the corresponding component in the provided array `ar`
   */
  def *+=(a1: Array[Float], ar: Array[Float]): Unit

  /**
   * Multiplies each component of '''this''' Tensor by the corresponding component in `a1`
   *  and subtracts the result from the corresponding component in the provided array `ar`
   */
  def *-=(a1: Array[Float], ar: Array[Float]): Unit

  /** Returns the result of component-wise addition of '''this''' Tensor with `t`*/
  def ++(t: Tensor1): Tensor1

  /** Apply the function `fn` to each component, passing its index and its value */
  def forEach(fn: (Int, Float) => Unit): Unit

  /** Returns this tensor as an Array of Floats */
  def asArray: Array[Float]
  
  def argmax: Int

  /** Returns the 1-norm of the Tensor1 */
  def norm1: Float

  /** Returns the 2-norm of the Tensor1 */
  def norm2: Float
  
  /** Map this tensor to an order-2 tensor with `r` rows */
  def toTensor2(r: Int) : Tensor2

  def copy: Tensor1

  def deepCopy: Tensor1

  def zeroOut(): Unit
  
  def mapInto(vv: Tensor1, fn: (Float, Float) => Float) =  {
    var i = 0; while (i < getDim) { update(i, fn(this(i), vv(i))); i += 1 }
  }
}

trait SparseTensor {
  def numNonZeros: Int
}

/**
 * Simple dense vector representation, wrapping a primitive `Array[Float]` but conforming 
 * to the `Tensor1` interface.
 * @param a Underlying array representing a dense vector
 * @param dr Density ratio
 * @author wellner
 */
class DenseTensor1(val a: Array[Float], dr: Float = 1.0f) extends Tensor1(dr) with Serializable {
  def this(nfs: Int) = this(Array.fill(nfs)(0.0f))
  def this(a: Array[Double]) = this(a map {_.toFloat})

  protected val size = a.length
  protected val dim = size

  def getSize = size
  def getDim = size

  @inline
  final def apply(i: Int) = a(i)

  @inline
  final def update(i: Int, v: Float) = { a(i) = v }

  def zeroOut() = {
	  var i = 0; while (i < size) {
		  a(i) = 0.0f
			i += 1
	  }
  }
  
  def *(v: Float) : Tensor1 = {
    new DenseTensor1(Array.tabulate(size)(i => a(i) * v))
  }
  
  def *(tt: Tensor1) : Float = {
	  tt match {
	  case t: DenseTensor1 =>
	    var r = 0.0f
	    var i = 0; while (i < dim) {
		    r += a(i) * t(i)
				i += 1
	    }
	    r
	  case t: SparseTensor1 => t * this
	  }
  }
  
  final def toTensor2(r: Int) : Tensor2 = {
    val ncols = size / r
    assert((size % r) == 0) 
    new DenseTensor2(a, r, ncols)
  }

  def numericalCheck() = {
    var i = 0; while (i < size) {
      if (a(i).isNaN()) println("***WARNING*** presence of NaN")
      if (a(i).isInfinite()) println("***WARNING*** presence of infinity")
      i += 1
    }
  }

  def copy = deepCopy

  def maxVal: Float = {
    var m = -Float.MaxValue
    var i = 0; while (i < dim) {
      if (a(i) > m) m = a(i)
      i += 1
    }
    m
  }

  def argmax: Int = {
    var m = -Float.MaxValue
    var mi = 0
    var i = 0; while (i < dim) {
      if (a(i) > m) {
        m = a(i)
        mi = i
      }
      i += 1
    }
    mi
  }

  override def toString() = {
    val sbuf = new StringBuilder
    sbuf append "Vector: "
    a foreach { v => sbuf append (" " + v) }
    sbuf append "\n"
    sbuf.toString()
  }

  def deepCopy = {
    val nArray = Array.tabulate(size) { i => a(i) }
    new DenseTensor1(nArray, dr)
  }
  
  def :=(vv: Tensor1) = {
    if (vv.getDim != this.getDim) throw new RuntimeException("Vector of length " + vv.getDim + " attempted on vector of length " + this.getDim)
    vv match {
      case v: DenseTensor1 => Array.copy(v.a, 0, a, 0, v.dim)
      case v: SparseTensor1 =>
        this.zeroOut()
        v.forEach({(i,v) => update(i,v)})
    }
    
  }

  def mapInPlace(fn: (Int,Float) => Float): Unit = {
    var i = 0; while (i < dim) { a(i) = fn(i,a(i)); i += 1 }
  }
   

  def map(fn: Float => Float): DenseTensor1 = {
    val na = Array.tabulate(dim) { i => fn(a(i)) }
    new DenseTensor1(na, dr)
  }
  
  
  final def :-(t: DenseTensor1) = new DenseTensor1(Array.tabulate(dim) { i => a(i) - t(i) })
  final def :+(t: DenseTensor1) = new DenseTensor1(Array.tabulate(dim) { i => a(i) + t(i) })
  final def :*(t: DenseTensor1) = new DenseTensor1(Array.tabulate(dim) { i => a(i) * t(i) })
  final def :/(t: DenseTensor1) = new DenseTensor1(Array.tabulate(dim) { i => a(i) / t(i) })

  final def +=(i: Int, v: Float) = a(i) += v
  final def -=(i: Int, v: Float) = a(i) -= v

  final def +=(v: Float) = { var i = 0; while (i < size) { a(i) += v; i += 1 } }
  final def *=(v: Float) = { var i = 0; while (i < size) { a(i) *= v; i += 1 } }
  final def -=(v: Float) = { var i = 0; while (i < size) { a(i) -= v; i += 1 } }
  final def :=(v: Float) = { var i = 0; while (i < size) { a(i) = v; i += 1 } }

  final def +=(t: Tensor1) = {
    t match {
      case dense: DenseTensor1 =>
        var i = 0; while (i < size) { a(i) += dense(i); i += 1 }
      case sparse: SparseTensor1 =>
        sparse forEach { (i, v) => a(i) += v }
    }
  }

  final def +=(t: Tensor1, mask: Array[Boolean]) = {
    t match {
      case dense: DenseTensor1 =>
        var i = 0; while (i < size) { if (mask(i)) a(i) += dense(i); i += 1 }
      case sparse: SparseTensor1 =>
        sparse forEach { (i, v) => if (mask(i)) a(i) += v }
    }
  }

  final def -=(t: Tensor1) = {
    t match {
      case dense: DenseTensor1 =>
        var i = 0; while (i < size) { a(i) -= dense(i); i += 1 }
      case sparse: SparseTensor1 =>
        sparse forEach { (i, v) => a(i) -= v }
    }
  }

  final def ^=(v: Float): Unit = { var i = 0; while (i < size) { Math.pow(a(i), v); i += 1 } }

  final def *+=(v: Float, ar: Array[Float]): Unit = {
    var i = 0; while (i < size) {
      ar(i) += a(i) * v
      i += 1
    }
  }

  final def *=(v: DenseTensor1) = {
    val dense = v.a
    var i = 0; while (i < size) { a(i) *= dense(i); i += 1 }
  }

  final def *-=(v: Float, ar: Array[Float]): Unit = *+=(-v, ar)

  final def *+=(a1: Array[Float], ar: Array[Float]): Unit = {
    var i = 0; while (i < size) {
      ar(i) += a(i) * a1(i)
      i += 1
    }
  }

  def *-=(a1: Array[Float], ar: Array[Float]): Unit = {
    var i = 0; while (i < size) {
      ar(i) -= a(i) * a1(i)
      i += 1
    }
  }

  def ++(t: Tensor1) = {
    this += t
    this
  }

  def forEach(fn: (Int, Float) => Unit): Unit = {
    var i = 0; while (i < size) {
      fn(i, a(i))
      i += 1
    }
  }

  def asArray = a

  def norm1: Float = {
    var n = 0.0f
    var i = 0; while (i < size) {
      n += math.abs(a(i))
      i += 1
    }
    n
  }

  def norm2: Float = {
    var n = 0.0
    var i = 0; while (i < size) { n += (a(i) * a(i)); i += 1 }
    math.sqrt(n.toDouble).toFloat  
  }

  def addMaskNoise(v: Float) = {
    var i = 0; while (i < size) { if (util.Random.nextFloat() < v) a(i) = 0.0f; i += 1 }
  }

}

object DenseTensor1 {
  def zeros(i: Int) = new DenseTensor1(i)
  def rand(i: Int) = tabulate(i) { _ => util.Random.nextFloat }
  def ones(i: Int) = {
    val a = Array.fill(i)(1.0f)
    new DenseTensor1(a)
  }
  def tabulate(i: Int)(fn: Int => Float) = {
    val a = Array.tabulate(i)(fn)
    new DenseTensor1(a)
  }
}

abstract class SparseTensor1(val dim: Int, dr: Float) extends Tensor1(dr) {
  protected val size = dim
  
  val indArray: Array[Int]
  val valArray: Array[Float]
  
  def numNonZeros : Int
  
  def getNonZeros : List[Int] = {
    var b : List[Int] = Nil
    forEach{(i,v) => if ((v > 0.0) || (v < 0.0)) b = i :: b}
    b
  }
  
  final def toTensor2(r: Int) : Tensor2 = {
    throw new RuntimeException("UNIMPLEMENTED Conversion to Sparse order-2 tensor")
  }
  
  def *(t: Float) : Tensor1 = {
    val nt = deepCopy
    nt.forEach({(i,v) => nt.update(i, v * t)})
    nt
  }
  
  def *(t: Tensor1) : Float = {
    var r = 0.0f
    forEach({(i,v) => r += v * t(i)})
    r
  }
  
  def getSize = dim
  def getDim = dim
  
  def norm1: Float = {
    var n = 0.0f
    forEach { (i, v) => n += math.abs(v) }
    n
  }

  def norm2: Float = {
    var n = 0.0f
    forEach { (i, v) => n += (v * v) }
    math.sqrt(n.toDouble).toFloat
  }
  
  def argmax = {
    var m = -1
    var mv = -Float.MaxValue
    this.forEach({(k,v) => if (v > mv) { mv = v; m = k}})
    m
  }

  final def :=(v: Float) = throw new RuntimeException("Assigning value to all components of a sparse tensor. Not allowed.")
  
  def *+=(v: Float, ar: Array[Float]): Unit = {
    forEach { (k, cv) =>
      val nv = ar(k) + (cv * v)
      ar.update(k, nv)
    }
  }

  def *-=(v: Float, ar: Array[Float]): Unit = {
    forEach { (k, cv) =>
      ar.update(k, (ar(k) - (cv * v)))
    }
  }

  def *+=(a1: Array[Float], ar: Array[Float]): Unit = forEach { (k, cv) => ar(k) += cv * a1(k) }

  def *-=(a1: Array[Float], ar: Array[Float]): Unit = forEach { (k, cv) => ar(k) -= cv * a1(k) }

  def convertToDense() = {
    val ar = Array.fill(dim)(0.0f)
    forEach { (i, v) => ar(i) = v }
    new DenseTensor1(ar)
  }

  def copy : SparseTensor1 = deepCopy // new SparseTensor1(dim, dr, umap)
  def deepCopy : SparseTensor1
  
  def asStatic : SparseTensor1
  

}

class StaticSparseTensor1(_dim: Int, dr: Float, val indArray: Array[Int], val valArray: Array[Float]) 
extends SparseTensor1(_dim, dr) with SparseTensor {
  
  val len = indArray.length
  
  def asStatic = this
  
  def numNonZeros : Int = indArray.length
  def deepCopy : SparseTensor1 = {    
    val ni = Array.tabulate(indArray.length)(i => indArray(i))
    val nv = Array.tabulate(valArray.length)(i => valArray(i))
    new StaticSparseTensor1(dim, dr, ni, nv)
  }
  
  def forEach(fn: ((Int, Float) => Unit)): Unit = {
    var j = 0; while (j < len) {
      val i = indArray(j)
      val v = valArray(j)
      fn(i,v)
      j += 1
    }
  }
  
  def apply(i: Int) = {
    var v = 0.0f
    var notFound = true
    var j = 0; while (notFound && (j < len)) {
      if (indArray(j) == i) { v = valArray(j); notFound = false }
      j += 1
    }
    v
  }
  
  def mapInPlace(fn: (Int,Float) => Float): Unit = {
    var i = 0; while (i < len) { 
      val vl = valArray(i)
      valArray(i) = fn(indArray(i),vl)
      i += 1 }
  }

  
  def :=(vv: Tensor1) = { throw new RuntimeException("Complex assignment not allowed with read-only SparseTensor1") }
  def +=(i: Int, v: Float) = { throw new RuntimeException("Complex assignment not allowed with read-only SparseTensor1") }
  def -=(i: Int, v: Float) = { throw new RuntimeException("Complex assignment not allowed with read-only SparseTensor1") }
  def +=(v: Float) = { throw new RuntimeException("Complex assignment not allowed with read-only SparseTensor1") }
  def *=(v: Float) = { throw new RuntimeException("Complex assignment not allowed with read-only SparseTensor1") }
  def +=(t: Tensor1) = { throw new RuntimeException("Complex assignment not allowed with read-only SparseTensor1") }
  def -=(t: Tensor1) = { throw new RuntimeException("Complex assignment not allowed with read-only SparseTensor1") }
  def ^=(v: Float): Unit = { throw new RuntimeException("Complex assignment not allowed with read-only SparseTensor1") }
  def asArray = { 
    val a = Array.fill(getDim)(0.0f)
    var i = 0; while (i < len) {
      a(indArray(i)) = valArray(i) 
      i += 1
    }
    a
  }
  
  def ++(t: Tensor1) = { throw new RuntimeException("Complex assignment not allowed with read-only SparseTensor1") }
  def zeroOut() = { throw new RuntimeException("Complex assignment not allowed with read-only SparseTensor1") }
  def update(i: Int, v: Float): Unit = { throw new RuntimeException("Complex assignment not allowed with read-only SparseTensor1") }
  
  override def toString() = {
    val sb = new StringBuilder
    asArray foreach {v => sb append " "; sb append v}
    sb.toString
  }
}

class DynamicSparseTensor1(_dim: Int, dr: Float = 0.1f, private val umap: OpenIntDoubleHashMap) 
extends SparseTensor1(_dim, dr) with SparseTensor {
  
  override def toString() = {
    val sb = new StringBuilder
    asArray foreach {v => sb append " "; sb append v}
    sb.toString
  }
  
  def asStatic = new StaticSparseTensor1(_dim, dr, indArray, valArray)
  
  lazy val indArray = {
    val indA = Array.fill(umap.size())(0)    
    var i = 0
    forEach{(ind,v) => indA(i) = ind; i += 1}
    indA
  }
  
  lazy val valArray : Array[Float] = {
    val valA = Array.fill(umap.size())(0.0f)
    var i = 0
    forEach{(ind,v) => valA(i) = v; i += 1}
    valA
  }
  
  def cacheArrays = {
    val a = indArray
    val b = valArray
    ()
  }
  
  
  def numNonZeros = umap.size()

  def zeroOut() = umap.clear()

  
  def deepCopy : SparseTensor1 = {
    val cc = new OpenIntDoubleHashMap
    this.forEach((i, v) => cc.put(i, v))
    new DynamicSparseTensor1(dim, dr, cc)
  }

  class ApplyFn(val fn: (Int, Float) => Unit) extends IntDoubleProcedure {
    def apply(k: Int, v: Float) = {
      fn(k, v)
      true
    }
    
    def apply(k: Int, v: Double) = apply(k, v.toFloat)
  }

  def forEach(fn: ((Int, Float) => Unit)): Unit = {
    umap.forEachPair(new ApplyFn(fn))
  }
  
  def apply(i: Int) = if (umap.containsKey(i)) umap.get(i).toFloat else 0.0f

  def mapInPlace(fn: (Int, Float) => Float): Unit = {
    var i = 0; while (i < dim) { this.update(i, fn(i,apply(i))); i += 1 }
  }

  def update(i: Int, v: Float): Unit = {
    umap.put(i, v)
  }

  def :=(vv: Tensor1) = {
    vv match {
      case v: SparseTensor1 =>
        umap.clear()
        this += v
      case x: DenseTensor1 => 
        umap.clear()
        var i = 0; while (i < x.getDim) {
          umap.put(i,x(i))
          i += 1
        }
    }    
  }
  
  def +=(i: Int, v: Float) = {
    val cv = if (umap.containsKey(i)) umap.get(i) else 0.0
    umap.put(i, cv + v)
  }

  def -=(i: Int, v: Float) = this += (i, -v)

  def +=(v: Float) = forEach { (k, cv) => umap.put(k, cv + v) }

  def *=(v: Float) = forEach { (k, cv) => umap.put(k, cv * v) }
  

  def +=(t: Tensor1) = t match {
    case dense: DenseTensor1 => throw new RuntimeException("Grave inefficiency: adding dense vector into static sparse vector")
    case sparse: SparseTensor1 => {
      sparse forEach { (i, v) =>
        val cv = if (umap.containsKey(i)) umap.get(i) else 0.0
        umap.put(i, v + cv)
      }
    }
  }
  
  def -=(t: Tensor1) = t match {
    case dense: DenseTensor1 => throw new RuntimeException("Grave inefficiency: adding dense vector into static sparse vector")
    case sparse: SparseTensor1 => {
      sparse forEach { (i, v) =>
        val cv = if (umap.containsKey(i)) umap.get(i) else 0.0
        umap.put(i, cv - v)  
      }
    }
  }

  def ^=(v: Float): Unit = forEach { (k, cv) => umap.put(k, Math.pow(cv, v)) }

  def asArray = convertToDense().asArray

  def ++(t: Tensor1) = {
    assert(this.dim == t.getDim)
    t match {
      case dense: DenseTensor1 =>
        forEach { (i, v) => dense += (i, v) }
        dense
      case sparse: SparseTensor1 =>
        val (larger, smaller) = if (sparse.numNonZeros > this.numNonZeros) (sparse, this) else (this, sparse)
        larger += smaller
        if ((densityRatio < 1.0) && ((larger.numNonZeros.toFloat / larger.getDim) > densityRatio))
          larger.convertToDense()
        else larger

    }
  }

  
}

object SparseTensor1 {
  def apply(dim: Int) = new DynamicSparseTensor1(dim, 0.1f, new OpenIntDoubleHashMap)
  def apply(dim: Int, dr: Float) = new DynamicSparseTensor1(dim, dr, new OpenIntDoubleHashMap)
  def apply(dim: Int, um: OpenIntDoubleHashMap) = new DynamicSparseTensor1(dim, 0.1f, um)
  def apply(dim: Int, dr: Float, um: OpenIntDoubleHashMap) = new DynamicSparseTensor1(dim, dr, um)
    
  def getOneHot(dim: Int, v: Int) = new StaticSparseTensor1(dim, 0.1f, Array(v), Array(1.0f))
  def getOnes(dim: Int, inds: Array[Int]) = new StaticSparseTensor1(dim, 0.1f, inds, Array.fill(inds.length)(1.0f))
  def getStaticSparseTensor1(dim: Int, inds: Array[Int], vls: Array[Float]) = new StaticSparseTensor1(dim, 0.1f, inds, vls)
}




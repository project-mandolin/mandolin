package org.mitre.mandolin.util
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import cern.colt.map.OpenIntDoubleHashMap
import cern.colt.function.IntDoubleProcedure

/**
 * Abstract top-level class for all tensor objects - sparse or dense.
 * Note that entries are always assumed to be Double types.
 * The class member densityRatio indicates the ratio at which sparse
 * representations are converted to dense (and possibly vice-versa).
 * A value of 1.0 indicates that sparse tensors will never be converted to 
 * dense representations.
 * 
 * @author Ben Wellner
 * @param densityRatio The ratio of non-zero entries to the size of the Tensor at 
 * which the `Tensor` will automatically switch to a dense representation.
 */
abstract class Tensor(val densityRatio: Double) extends Serializable {

  protected val size: Int
  def getSize: Int

  /** IndexedSeq with the size of each dimension */
  def getDimSizes: collection.immutable.IndexedSeq[Int]
}

/**
 *  Represents a 1-dimensional vector
 */
abstract class Tensor1(dr: Double = 1.0) extends Tensor(dr) {

  def getDimSizes = collection.immutable.IndexedSeq(getDim)
  
  
  
  /** Assign values in tensor `v` to this `Tensor1` */
  def :=(v: Tensor1) : Unit
  
  /** Assign value `v` to each component in `Tensor1` */
  def :=(v: Double) : Unit
  
  /** Gets the number of dimensions for this `Tensor1` - i.e. vector */
  def getDim: Int

  /** Access the ''i''th element of the `Tensor1` object */
  def apply(i: Int): Double

  /** Update the ''i''th element of the `Tensor1` object */
  def update(i: Int, v: Double): Unit

  def mapInPlace(fn: (Double => Double)): Unit
  
  /** Dot product */
  def *(t: Tensor1) : Double
  
  /** Matrix-scalar product. Returns new `Tensor1` */
  def *(v: Double) : Tensor1

  /** Add the value `v` to the ''i''th element of the Tensor */
  def +=(i: Int, v: Double): Unit

  /** Add the value `v` to the ''i''th element of the Tensor */
  def -=(i: Int, v: Double): Unit

  /** Adds the value `v` to all */
  def +=(v: Double): Unit

  /** Multiply each component of the Tensor by `v` */
  def *=(v: Double): Unit

  /** Divide each component of the Tensor by `v` */
  def /=(v: Double): Unit = *=(1.0 / v)

  /** Component-wise addition */
  def +=(t: Tensor1): Unit

  /** Component-wise addition */
  def -=(t: Tensor1): Unit
  
  /** Raise each element of this tensor to a power */
  def ^=(v: Double): Unit

  /**
   * Multiplies each component of '''this''' Tensor by `v` and adds it to the corresponding component
   *  in the provided array `ar`
   */
  def *+=(v: Double, ar: Array[Double]): Unit

  /**
   * Takes each component of '''this''' Tensor and adds it to the corresponding component
   *  in the provided array `ar`
   */
  def +=(ar: Array[Double]): Unit = *+=(1.0, ar)

  /**
   * Multiplies each component of '''this''' Tensor by `v` and subtracts it from the corresponding component
   *  in the provided array `ar`
   */
  def *-=(v: Double, ar: Array[Double]): Unit

  /**
   * Multiplies each component of '''this''' Tensor by the corresponding component in `a1`
   *  and adds the result to the corresponding component in the provided array `ar`
   */
  def *+=(a1: Array[Double], ar: Array[Double]): Unit

  /**
   * Multiplies each component of '''this''' Tensor by the corresponding component in `a1`
   *  and subtracts the result from the corresponding component in the provided array `ar`
   */
  def *-=(a1: Array[Double], ar: Array[Double]): Unit

  /** Returns the result of component-wise addition of '''this''' Tensor with `t`*/
  def ++(t: Tensor1): Tensor1

  /** Apply the function `fn` to each component, passing its index and its value */
  def forEach(fn: (Int, Double) => Unit): Unit

  /** Returns this tensor as an Array of Doubles */
  def asArray: Array[Double]
  
  def argmax: Int

  /** Returns the 1-norm of the Tensor1 */
  def norm1: Double

  /** Returns the 2-norm of the Tensor1 */
  def norm2: Double
  
  /** Map this tensor to an order-2 tensor with `r` rows */
  def toTensor2(r: Int) : Tensor2

  def copy: Tensor1

  def deepCopy: Tensor1

  def zeroOut(): Unit
  
  def mapInto(vv: Tensor1, fn: (Double, Double) => Double) =  {
    var i = 0; while (i < getDim) { update(i, fn(this(i), vv(i))); i += 1 }
  }
}

trait SparseTensor {
  def numNonZeros: Int
}

/**
 * Simple dense vector representation, wrapping a primitive `Array[Double]` but conforming 
 * to the `Tensor1` interface.
 * @param a Underlying array representing a dense vector
 * @param dr Density ratio
 * @author wellner
 */
class DenseTensor1(val a: Array[Double], dr: Double = 1.0) extends Tensor1(dr) with Serializable {
  def this(nfs: Int) = this(Array.fill(nfs)(0.0))

  protected val size = a.length
  protected val dim = size

  def getSize = size
  def getDim = size

  @inline
  final def apply(i: Int) = a(i)

  @inline
  final def update(i: Int, v: Double) = { a(i) = v }

  def zeroOut() = {
	  var i = 0; while (i < size) {
		  a(i) = 0.0
			i += 1
	  }
  }
  
  def *(v: Double) : Tensor1 = {
    new DenseTensor1(Array.tabulate(size)(i => a(i) * v))
  }
  
  def *(tt: Tensor1) : Double = {
	  tt match {
	  case t: DenseTensor1 =>
	    var r = 0.0
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

  def maxVal: Double = {
    var m = -Double.MaxValue
    var i = 0; while (i < dim) {
      if (a(i) > m) m = a(i)
      i += 1
    }
    m
  }

  def argmax: Int = {
    var m = -Double.MaxValue
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
    assert(vv.getDim == this.getDim)
    vv match {
      case v: DenseTensor1 => Array.copy(v.a, 0, a, 0, v.dim)
      case v: SparseTensor1 =>
        this.zeroOut()
        v.forEach({(i,v) => update(i,v)})
    }
    
  }

  def mapInPlace(fn: Double => Double): Unit = {
    var i = 0; while (i < dim) { a(i) = fn(a(i)); i += 1 }
  }

  def map(fn: Double => Double): DenseTensor1 = {
    val na = Array.tabulate(dim) { i => fn(a(i)) }
    new DenseTensor1(na, dr)
  }
  
  
  final def :-(t: DenseTensor1) = new DenseTensor1(Array.tabulate(dim) { i => a(i) - t(i) })
  final def :+(t: DenseTensor1) = new DenseTensor1(Array.tabulate(dim) { i => a(i) + t(i) })
  final def :*(t: DenseTensor1) = new DenseTensor1(Array.tabulate(dim) { i => a(i) * t(i) })
  final def :/(t: DenseTensor1) = new DenseTensor1(Array.tabulate(dim) { i => a(i) / t(i) })

  final def +=(i: Int, v: Double) = a(i) += v
  final def -=(i: Int, v: Double) = a(i) -= v

  final def +=(v: Double) = { var i = 0; while (i < size) { a(i) += v; i += 1 } }
  final def *=(v: Double) = { var i = 0; while (i < size) { a(i) *= v; i += 1 } }
  final def -=(v: Double) = { var i = 0; while (i < size) { a(i) -= v; i += 1 } }
  final def :=(v: Double) = { var i = 0; while (i < size) { a(i) = v; i += 1 } }

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

  final def ^=(v: Double): Unit = { var i = 0; while (i < size) { Math.pow(a(i), v); i += 1 } }

  final def *+=(v: Double, ar: Array[Double]): Unit = {
    var i = 0; while (i < size) {
      ar(i) += a(i) * v
      i += 1
    }
  }

  final def *=(v: DenseTensor1) = {
    val dense = v.a
    var i = 0; while (i < size) { a(i) *= dense(i); i += 1 }
  }

  final def *-=(v: Double, ar: Array[Double]): Unit = *+=(-v, ar)

  final def *+=(a1: Array[Double], ar: Array[Double]): Unit = {
    var i = 0; while (i < size) {
      ar(i) += a(i) * a1(i)
      i += 1
    }
  }

  def *-=(a1: Array[Double], ar: Array[Double]): Unit = {
    var i = 0; while (i < size) {
      ar(i) -= a(i) * a1(i)
      i += 1
    }
  }

  def ++(t: Tensor1) = {
    this += t
    this
  }

  def forEach(fn: (Int, Double) => Unit): Unit = {
    var i = 0; while (i < size) {
      fn(i, a(i))
      i += 1
    }
  }

  def asArray = a

  def norm1: Double = {
    var n = 0.0
    var i = 0; while (i < size) {
      n += math.abs(a(i))
      i += 1
    }
    n
  }

  def norm2: Double = {
    var n = 0.0
    var i = 0; while (i < size) { n += (a(i) * a(i)); i += 1 }
    math.sqrt(n)
  }

  def addMaskNoise(v: Double) = {
    var i = 0; while (i < size) { if (util.Random.nextDouble() < v) a(i) = 0.0; i += 1 }
  }

}

object DenseTensor1 {
  def zeros(i: Int) = new DenseTensor1(i)
  def rand(i: Int) = tabulate(i) { _ => util.Random.nextDouble }
  def ones(i: Int) = {
    val a = Array.fill(i)(1.0)
    new DenseTensor1(a)
  }
  def tabulate(i: Int)(fn: Int => Double) = {
    val a = Array.tabulate(i)(fn)
    new DenseTensor1(a)
  }
}

class SparseTensor1(val dim: Int, dr: Double = 0.1, private val umap: OpenIntDoubleHashMap) extends Tensor1(dr) with SparseTensor {
  protected val size = dim
  
  lazy val indArray = {
    val indA = Array.fill(umap.size())(0)    
    var i = 0
    forEach{(ind,v) => indA(i) = ind; i += 1}
    indA
  }
  
  lazy val valArray = {
    val valA = Array.fill(umap.size())(0.0)
    var i = 0
    forEach{(ind,v) => valA(i) = v; i += 1}
    valA
  }
  
  def cacheArrays = {
    val a = indArray
    val b = valArray
    ()
  }
  
  def getNonZeros : List[Int] = {
    var b : List[Int] = Nil
    forEach{(i,_) => b = i :: b}
    b
  }
  
  final def toTensor2(r: Int) : Tensor2 = {
    throw new RuntimeException("UNIMPLEMENTED Conversion to Sparse order-2 tensor")
  }
  
  def *(t: Double) : Tensor1 = {
    val nt = deepCopy
    nt.forEach({(i,v) => nt.update(i, v * t)})
    nt
  }
  
  def *(t: Tensor1) : Double = {
    var r = 0.0
    forEach({(i,v) => r += v * t(i)})
    r
  }
  
  def getSize = dim
  def getDim = dim

  def numNonZeros = umap.size()

  def zeroOut() = umap.clear()

  def copy = deepCopy // new SparseTensor1(dim, dr, umap)
  def deepCopy = {
    val cc = new OpenIntDoubleHashMap
    this.forEach((i, v) => cc.put(i, v))
    new SparseTensor1(dim, dr, cc)
  }

  class ApplyFn(val fn: (Int, Double) => Unit) extends IntDoubleProcedure {
    def apply(k: Int, v: Double) = {
      fn(k, v)
      true
    }
  }

  def forEach(fn: ((Int, Double) => Unit)): Unit = {
    umap.forEachPair(new ApplyFn(fn))
  }
  
  def argmax = {
    var m = -1
    var mv = -Double.MaxValue
    this.forEach({(k,v) => if (v > mv) { mv = v; m = k}})
    m
  }

  def apply(i: Int) = if (umap.containsKey(i)) umap.get(i) else 0.0

  def mapInPlace(fn: Double => Double): Unit = {
    var i = 0; while (i < dim) { this.update(i, fn(apply(i))); i += 1 }
  }

  def update(i: Int, v: Double): Unit = {
    umap.put(i, v)
  }

  def :=(vv: Tensor1) = {
    vv match {
      case v: SparseTensor1 =>
        umap.clear()
        this += v
      case _: DenseTensor1 => throw new RuntimeException("Assigning dense tensor to sparse tensor variable. Not allowed.")
    }    
  }
  
  final def :=(v: Double) = throw new RuntimeException("Assigning value to all components of a sparse tensor. Not allowed.")

  def +=(i: Int, v: Double) = {
    val cv = if (umap.containsKey(i)) umap.get(i) else 0.0
    umap.put(i, cv + v)
  }

  def -=(i: Int, v: Double) = this += (i, -v)

  def +=(v: Double) = forEach { (k, cv) => umap.put(k, cv + v) }

  def *=(v: Double) = forEach { (k, cv) => umap.put(k, cv * v) }
  

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
        umap.put(i, v - cv)
      }
    }
  }

  def ^=(v: Double): Unit = forEach { (k, cv) => umap.put(k, Math.pow(cv, v)) }

  def *+=(v: Double, ar: Array[Double]): Unit = {
    forEach { (k, cv) =>
      val nv = ar(k) + (cv * v)
      ar.update(k, nv)
    }
  }

  def *-=(v: Double, ar: Array[Double]): Unit = {
    forEach { (k, cv) =>
      ar.update(k, (ar(k) - (cv * v)))
    }
  }

  def *+=(a1: Array[Double], ar: Array[Double]): Unit = forEach { (k, cv) => ar(k) += cv * a1(k) }

  def *-=(a1: Array[Double], ar: Array[Double]): Unit = forEach { (k, cv) => ar(k) -= cv * a1(k) }

  def convertToDense() = {
    val ar = Array.fill(dim)(0.0)
    forEach { (i, v) => ar(i) = v }
    new DenseTensor1(ar)
  }

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
        if ((densityRatio < 1.0) && ((larger.numNonZeros.toDouble / larger.getDim) > densityRatio))
          larger.convertToDense()
        else larger

    }
  }

  def norm1: Double = {
    var n = 0.0
    forEach { (i, v) => n += math.abs(v) }
    n
  }

  def norm2: Double = {
    var n = 0.0
    forEach { (i, v) => n += (v * v) }
    math.sqrt(n)
  }
}

object SparseTensor1 {
  def apply(dim: Int) = new SparseTensor1(dim, 0.1, new OpenIntDoubleHashMap)
  def apply(dim: Int, dr: Double) = new SparseTensor1(dim, dr, new OpenIntDoubleHashMap)
  def apply(dim: Int, um: OpenIntDoubleHashMap) = new SparseTensor1(dim, 0.1, um)
  def apply(dim: Int, dr: Double, um: OpenIntDoubleHashMap) = new SparseTensor1(dim, dr, um)
}




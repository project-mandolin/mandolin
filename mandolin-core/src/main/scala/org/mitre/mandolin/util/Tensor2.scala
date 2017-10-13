package org.mitre.mandolin.util
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import collection.mutable.HashMap

/**
 * 
 * Represents a 2-dimensional tensor - i.e. a matrix
 * @param dr - density ratio used to switch representation between sparse and dense
 * @author wellner
 */
abstract class Tensor2(dr: Float = 1.0f) extends Tensor(dr) with Serializable {
  def getDim1: Int
  def getDim2: Int
  def apply(i: Int, j: Int): Float
  def update(i: Int, j: Int, v: Float): Unit
  def getRow(i: Int): Tensor1
  def getCol(i: Int): Tensor1
  def *=(v: Float): Unit
  def +=(v: Float): Unit
  def -=(v: Float): Unit
  def /=(v: Float): Unit
  def :=(v: Float): Unit
  def +=(m: Tensor2): Unit
  
  /** take a row from this matrix and compute dot product with provided vector */
  def rowDot(row: Int, v: Tensor1) : Float
  
  /** Zero out matrix */
  def clear() : Unit
  
  /*
   * Perform a component-wise addition of vector terms to each row of the tensor
   */
  def +=(vv: Tensor1) : Unit
  
  def getDimSizes = collection.immutable.IndexedSeq(getDim1, getDim2)

  def *=(vv: Tensor1, res: DenseTensor1): Unit
  def *=(vv: Tensor1, res: DenseTensor1, m: Array[Boolean]): Unit

  def trMult(vv: DenseTensor1, res: DenseTensor1): Unit

  def copy(): Tensor2

  def mapInto(m: Tensor2, fn: (Float, Float) => Float): Unit

  def outerFill(v1: DenseTensor1, v2: SparseTensor1): Unit
  def outerFill(v1: DenseTensor1, v2: DenseTensor1): Unit

  def outerFill(v1: DenseTensor1, v2: Tensor1): Unit = {
    v2 match { 
      case x: SparseTensor1 => outerFill(v1, x) 
      case x: DenseTensor1 => outerFill(v1, x) }
  }
  
  def asArray : Array[Float]
  
  def transpose : Tensor2

}

/**
 * @param a - underlying array representation of matrix
 * @param nrows - number of rows
 * @param ncols - number of columns
 * @param dr - density ratio 
 * @author 
 */
class DenseTensor2(val a: Array[Float], val nrows: Int, val ncols: Int, dr: Float = 1.0f) extends Tensor2(dr) with Serializable {
  def this(r: Int, c: Int) = this(Array.fill(r * c)(0.0f), r, c)

  protected val size = a.length
  
  def getSize = size
  def getDim1 = nrows
  def getDim2 = ncols

  @inline final def apply(i: Int, j: Int): Float = a(i * ncols + j)
  @inline final def update(i: Int, j: Int, v: Float) = a(i * ncols + j) = v
  @inline final def getRow(i: Int): Tensor1 = new DenseTensor1(Array.tabulate(ncols)(j => a(i * ncols + j)))
  @inline final def getCol(i: Int): Tensor1 = new DenseTensor1(Array.tabulate(nrows)(j => a(j * ncols + i)))
  
  def asArray = a
  
  def clear() = {
    var i = 0; while (i < size) {
      a(i) = 0.0f
      i += 1
    }
  }
  
  override def toString() = {
    val sb = new StringBuilder
    var i = 0; while (i < size) {
      if ((i % ncols) == 0) sb append '\n'
      sb append a(i)
      sb append ' '      
      i += 1
    }
    sb append '\n'
    sb.toString
  }
 
  def copy(): Tensor2 = {
    val na = Array.tabulate(nrows * ncols)(i => a(i))
    new DenseTensor2(na, nrows, ncols, dr)
  }
  
  def transpose : Tensor2 = {
    val na = Array.fill(nrows * ncols)(0.0f)
    val nnr = ncols
    val nnc = nrows
    val trans = new DenseTensor2(na, nnr, nnc)
    var i = 0; while (i < nrows) {
      var j = 0; while (j < ncols) {
        trans(j,i) = this(i,j)
        j += 1
      }
      i += 1
    }
    trans
  }
  
  final def rowDot(row: Int, v: Tensor1) = {
    var r = 0.0f
    v match {
      case x: DenseTensor1 =>
        val l = x.getDim
        val offset = row * ncols // this may be faster...
        var i = 0; while (i < l) {
          r += a(offset + i) * x(i)
          i += 1
        }        
      case x: SparseTensor1 => throw new RuntimeException("Sparse tensor unsupported against dense matrix with rotDot method")
    }
    r
  }

  final def *=(v: Float) = { var i = 0; while (i < size) { a(i) *= v; i += 1 } }
  final def +=(v: Float) = { var i = 0; while (i < size) { a(i) += v; i += 1 } }
  final def -=(v: Float) = { var i = 0; while (i < size) { a(i) -= v; i += 1 } }
  final def /=(v: Float) = { var i = 0; while (i < size) { a(i) /= v; i += 1 } }
  final def :=(v: Float) = { var i = 0; while (i < size) { a(i) = v; i += 1 } }
  final def :=(dm: DenseTensor2) = {
    System.arraycopy(dm, 0, a, 0, size)
  }
  
  final def +=(vv: Tensor1) : Unit = {
    var i = 0; while (i < nrows) {
      var j = 0; while (j < ncols) {
        a(i * ncols + j) += vv(i)        
        j += 1
      }
      i += 1
    }
  }

  final def +=(mm: Tensor2) = mm match {
    case m: DenseTensor2 =>
      var i = 0; while (i < size) { a(i) += m.a(i); i += 1 }
    case m: ColumnSparseTensor2 =>
      var i = 0
      while (i < nrows) {
        val row = m.a(i)
        row.forEach { (j, v) =>
          val cv = this(i, j)
          this(i, j) = cv + v
        }
        i += 1
      }
    case m: RowSparseTensor2 =>
      m.forEachRow({(i,row) =>
        var j = 0; while (j < ncols) {
          val cv = this(i,j)
          this(i,j) = cv + row(j)
          j += 1
        }
        })
  }

  def mapInto(mm: Tensor2, fn: (Float, Float) => Float) = mm match {
    case m: DenseTensor2 =>
      var i = 0; while (i < size) { a(i) = fn(a(i), m.a(i)); i += 1 }
    case m: ColumnSparseTensor2 =>
      var i = 0; while (i < nrows) {
        val mr = m.a(i)
        mr.forEach { (j, v) => this(i, j) = fn(this(i, j), v) }
        i += 1
      }
    case m: RowSparseTensor2 =>
      m.forEachRow({(i,row) =>
        var j = 0; while (j < ncols) {
          this(i,j) = fn(this(i,j), row(j))
          j += 1
        }
        })
  }

  /** Multiply this matrix with a vector `vv`, inserting the result into `res` */
  def *=(vv: Tensor1, res: DenseTensor1): Unit = {
    vv match {
      case x: DenseTensor1  => this *= (x, res)
      case x: SparseTensor1 => this *= (x, res)
    }
  }

  /** Multiply this matrix with a vector `vv`, inserting the result into the dense vector `res`, 
   *  masking entries according to the bitvector `m` */
  def *=(vv: Tensor1, res: DenseTensor1, m: Array[Boolean]): Unit = {
    vv match {
      case x: DenseTensor1  => this *= (x, res, m)
      case x: SparseTensor1 => this *= (x, res, m)
    }
  }

  /** Multiple this matrix with vector `vv`, inserting result into dense vector `res` */
  def *=(vv: DenseTensor1, res: DenseTensor1): Unit = {
    var i = 0 // iterate over rows
    val v = vv.a // get underlying array
    val r = res.a
    while (i < nrows) {
      // get dot product
      var dp = 0.0f
      var j = 0
      val offset = ncols * i
      while (j < ncols) {
        dp += a(offset + j) * v(j)
        j += 1
      }
      r(i) = dp
      i += 1
    }
  }

  /** Multiple this matrix with ''sparse'' vector `vv`, inserting result into dense vector `res` */
  def *=(vv: SparseTensor1, res: DenseTensor1): Unit = {
    var i = 0 // iterate over rows
    val r = res.a
    while (i < nrows) {
      var dp = 0.0f
      val offset = ncols * i
      vv.forEach({ (j, v) => dp += v * a(offset + j) })
      r(i) = dp
      i += 1
    }
  }
  
  /** Multiple this matrix with ''sparse'' vector `vv`, inserting result into dense vector `res`
   *  while masking based on bit-vector `m` */
  def *=(vv: SparseTensor1, res: DenseTensor1, mask: Array[Boolean]): Unit = {
    var i = 0
    val r = res.a
    while (i < nrows) {
      if (mask(i)) {
        var dp = 0.0f
        val offset = ncols * i
        vv.forEach({ (j, v) => dp += v * a(offset + j) })
        r(i) = dp
      } else res(i) = 0.0f
      i += 1
    }
  }

  /** Multiple this matrix with ''dense'' vector `vv`, inserting result into dense vector `res`
   *  while masking based on bit-vector `m` */
  def *=(vv: DenseTensor1, res: DenseTensor1, mask: Array[Boolean]) = {
    var i = 0 // iterate over rows
    val v = vv.a
    val r = res.a
    while (i < nrows) {
      if (mask(i)) {
        var dp = 0.0f
        var j = 0
        val offset = ncols * i
        while (j < ncols) {
          dp += a(offset + j) * v(j)
          j += 1
        }
        r(i) = dp
      } else res(i) = 0.0f
      i += 1
    }
  }

  /** Multiply transpose of this matrix by ''dense'' vector `vv`, placing result in `res` */
  def trMult(vv: DenseTensor1, res: DenseTensor1): Unit = {
    var i = 0
    val v = vv.a
    val r = res.a
    while (i < ncols) {
      var dp = 0.0f
      var j = 0
      while (j < nrows) {
        dp += a(ncols * j + i) * v(j)
        j += 1
      }
      r(i) = dp
      i += 1
    }
  }


  /** Perform an outer product of `v1` and `v2` where both are ''dense'' and place
   *  result in `this` matrix */
  def outerFill(v1: DenseTensor1, v2: DenseTensor1): Unit = {
    var i = 0; while (i < nrows) {
      var j = 0; while (j < ncols) {
        a(i * ncols + j) = v1(i) * v2(j)
        j += 1
      }
      i += 1
    }
  }

  /** Perform an outer product of `v1` and `v2` where v1 is ''dense'' and 
   *  v2 is ''sparse''; place result in `this` matrix */
  def outerFill(v1: DenseTensor1, v2: SparseTensor1): Unit = {
    var i = 0; while (i < nrows) {
      v2.forEach({ (k, v) => a(i * ncols + k) = v1(i) * v })
      i += 1
    }
  }

}

class Simple2DArray(val ars: Array[Array[Float]], val nrows: Int, val ncols: Int) {
  
  def apply(i: Int, j: Int) = ars(i)(j)
  def update(i: Int, j: Int, v: Float) = ars(i)(j) = v
  def getDim1 = nrows
  def getDim2 = ncols
  def +=(i: Int, j: Int, v: Float) = ars(i)(j) += v 
}

class Simple2DByteArray(val ars: Array[Array[Byte]], val nrows: Int, val ncols: Int) {
  
  def apply(i: Int, j: Int) = ars(i)(j)
  def update(i: Int, j: Int, v: Byte) = ars(i)(j) = v
  def getDim1 = nrows
  def getDim2 = ncols

}


object Simple2DArray {
    
  def floatArray(i: Int,j: Int) = new Simple2DArray(Array.tabulate(i){_ => Array.fill(j)(0.0f)}, i, j)
  
  def byteArray(i: Int, j: Int) = new Simple2DByteArray(Array.tabulate(i){_ => Array.fill(j)(0)}, i, j)
}


/**
 * A matrix (tensor2) with sparse columns. Because matrix is sparse, many operations available for dense matrices aren't
 * provided as they would result in a dense matrix.
 * @param a - underlying array of sparse vectors
 * @param nrows - number of rows
 * @param ncols - number of columns
 * @param dr - density ratio (if density exceeds this ratio, tensor should convert to dense representation)
 */
class ColumnSparseTensor2(val a: Array[SparseTensor1], val nrows: Int, val ncols: Int, dr: Float = 1.0f) extends Tensor2(dr) with Serializable {
  def this(r: Int, c: Int) = this(Array.tabulate(r)(ri => SparseTensor1(c)), r, c)

  protected val size = a(0).getDim * a.length
  
  def getSize = size
  def getDim1 = nrows
  def getDim2 = ncols
  
  @inline final def apply(i: Int) = a(i)

  @inline final def apply(i: Int, j: Int): Float = a(i)(j)
  @inline final def update(i: Int, j: Int, v: Float) = {
    val row = a(i)
    row.update(j,v)
  }
  @inline final def getRow(i: Int): Tensor1 = a(i)
  @inline final def getCol(i: Int): Tensor1 = throw new RuntimeException("Column extraction for Column Sparse matrix not implemented")
  @inline final def :=(v: Float) =
    if (v == 0.0) {
      var i = 0; while (i < nrows) { a(i).zeroOut(); i += 1 }
    } else throw new RuntimeException("Non-zero value assigned to entire sparse matrix. Not allowed.")

  final def -=(v: Float) = throw new RuntimeException("Non-zero value assigned to entire sparse matrix. Not allowed.")
  final def +=(v: Float) = throw new RuntimeException("Non-zero value assigned to entire sparse matrix. Not allowed.")
  
  final def +=(vv: Tensor1) : Unit = throw new RuntimeException("Operation results in non-sparse Matrix") 
  final def rowDot(row: Int, vec: Tensor1) = {
    var r = 0.0f
    vec match {
      case x: DenseTensor1 => 
        val spRow = a(row)
        spRow.forEach({(i,v) => r += v * vec(i)})
      case _: SparseTensor1 => throw new RuntimeException("Sparse vector not allows in dot product with row of matrix.")
    }
    r
  }
  
  def transpose : Tensor2 = throw new RuntimeException("Transpose not supported for column sparse tensors")
  
  def asArray = {
      //throw new RuntimeException("As array not possible with ColumnSparseTensor2 having " + a.length + " rows")
      val ar = Array.fill(size)(0.0f)
      var r = 0; while (r < a.length) {
        val row = a(r)
        row.forEach((c,v) => ar(c + (ncols * r)) = v)
        r += 1
      }
      ar
    }

  /** This performs a SPARSE map into.  Only non-zero elements in THIS matrix are updated */
  def mapInto(mm: Tensor2, fn: (Float, Float) => Float) = mm match {
    case m: ColumnSparseTensor2 =>
      var i = 0; while (i < nrows) {
        val row = a(i)
        row.forEach { (j, v) => row(j) = fn(m(i, j), v) }
        i += 1
      }
    case _ => throw new RuntimeException("Sparse/dense matrix incompatibility")
  }
  
  def trMult(vv: DenseTensor1, res: DenseTensor1): Unit = {
    throw new RuntimeException("Transpose matrix product inefficient with column-sparse representation")
  }

  def copy(): Tensor2 = {
    val na = Array.tabulate(nrows)(i => a(i).copy)
    new ColumnSparseTensor2(na, nrows, ncols, dr)
  }

  final def *=(v: Float) = { var i = 0; while (i < nrows) { a(i) *= v; i += 1 } }
  final def /=(v: Float) = { var i = 0; while (i < nrows) { a(i) /= v; i += 1 } }
  final def +=(mm: Tensor2) = mm match {
    case m: DenseTensor2 => throw new RuntimeException("Adding dense tensor to sparse .. failure")
    case m: ColumnSparseTensor2 =>
      var i = 0
      while (i < nrows) {
        a(i) += m.a(i)
        i += 1
      }
  }

  def *=(vv: Tensor1, res: DenseTensor1): Unit = {
    var i = 0 // iterate over rows
    val r = res.a
    while (i < nrows) {
      r(i) = a(i) * vv
      i += 1
    }
  }
  
  def *=(vv: Tensor1, res: DenseTensor1, mask: Array[Boolean]): Unit = {
    var i = 0
    val r = res.a
    while (i < nrows) {
      if (mask(i)) {
        r(i) = a(i) * vv
      } else r(i) = 0.0f
      i += 1
    }
  }


  /** Perform an outer product of `v1` and `v2` where v1 is ''dense'' and 
   *  v2 is ''sparse''; place sparse result in `this` matrix */
  def outerFill(v1: DenseTensor1, v2: SparseTensor1): Unit = {
    var i = 0; while (i < nrows) {
      val row = a(i)      
      if (row.numNonZeros > 0) row.zeroOut() // zero out since these are sparse updates
      v2.forEach({ (k, v) => row.update(k, v1(i) * v) })
      i += 1
    }
  }

  def outerFill(v1: DenseTensor1, v2: DenseTensor1): Unit = throw new RuntimeException("Sparse tensor cannot accept dense outer product")
  
  override def toString() = {
    val sbuf = new StringBuilder
    sbuf append "Vector: "
    a(0).forEach { (i,v) => sbuf append (" i: " + v) }
    sbuf append "\n"
    sbuf.toString()
  }
  
  def clear() = {
    var i = 0; while (i < getDim1) {
      a(i).zeroOut() // clear each row      
      i += 1
    }
  }
}

/**
 * A matrix (tensor2) with sparse rows. Because matrix is sparse, many operations available for dense matrices aren't
 * provided as they would result in a dense matrix.
 * @param rows - sparse array of dense vectors
 * @param nrows - number of rows
 * @param ncols - number of columns
 * @param dr - density ratio (if density exceeds this ratio, tensor should convert to dense representation)
 */
class RowSparseTensor2(val rows: HashMap[Int,Tensor1], val nrows: Int, val ncols: Int, dr: Float = 1.0f) 
extends Tensor2(dr) with Serializable {

  protected val size = nrows * ncols
  def getSize = size
  def getDim1 = nrows
  def getDim2 = ncols
  
  def trMult(vv: DenseTensor1, res: DenseTensor1): Unit = {
    throw new RuntimeException("Transpose matrix product inefficient with row-sparse representation")
  }
  
  @inline final def apply(i: Int, j: Int): Float = rows(i)(j)
  @inline final def update(i: Int, j: Int, v: Float) = {
    val row = rows(i)
    row.update(i,v)
  }
  @inline final def getRow(i: Int): Tensor1 = rows(i)
  @inline final def getCol(i: Int): Tensor1 = throw new RuntimeException("Column extraction for Column Sparse matrix not implemented")
  @inline final def :=(v: Float) =
    if (v == 0.0) {
      var i = 0; while (i < nrows) { rows(i).zeroOut(); i += 1 }
    } else throw new RuntimeException("Non-zero value assigned to entire sparse matrix. Not allowed.")

  final def -=(v: Float) = throw new RuntimeException("Non-zero value assigned to entire sparse matrix. Not allowed.")
  final def +=(v: Float) = throw new RuntimeException("Non-zero value assigned to entire sparse matrix. Not allowed.")
  
  final def +=(vv: Tensor1) : Unit = throw new RuntimeException("Operation results in non-sparse Matrix") 
  final def rowDot(row: Int, vec: Tensor1) = {
    rows(row) * vec
  }
  
  override def toString() = {
    val asList = rows.toList.sortBy(x => x._1)
    val sb = new StringBuilder
    asList foreach {case (i,v) =>
      val aa = v.asArray
      var j = 0; while (j < aa.length) {
        if (j > 0) sb append " "
        sb append (aa(j))
        j += 1
      }
      sb append "\n"
      }
    sb.toString
  }
  
  def forEachRow(fn: (Int, Tensor1) => Unit) = {
    rows foreach {v => fn(v._1,v._2)}
  }
  
  def asArray = {
      throw new RuntimeException("As array not possible with RowSparseTensor2 having sparse rows")      
    }
  
  def transpose : Tensor2 = throw new RuntimeException("Transpose not supported for row sparse tensors")

  /** This performs a SPARSE map into.  Only non-zero elements in THIS matrix are updated */
  def mapInto(mm: Tensor2, fn: (Float, Float) => Float) = mm match {
    case _ => throw new RuntimeException("Sparse/dense matrix incompatibility")
  }
  
  def copy(): Tensor2 = {
    val na = rows map {case (i,r) => (i, r.copy)}
    new RowSparseTensor2(na, nrows, ncols)
  }

  final def *=(v: Float) = rows foreach {case (i,r) => r *= v}
  final def /=(v: Float) = rows foreach {case (i,r) => r /= v}
  final def +=(mm: Tensor2) = mm match {
    case _ => throw new RuntimeException("Adding dense tensor to sparse .. failure")    
  }

  def *=(vv: Tensor1, res: DenseTensor1): Unit = {
    rows foreach {case (i, row) => res.update(i, row * vv) }
  }
  
  def *=(vv: Tensor1, res: DenseTensor1, mask: Array[Boolean]): Unit = {
    rows foreach {case (i, row) => if (mask(i)) res.update(i, row * vv) }    
  }

  /** Perform an outer product of `v1` and `v2` where v1 is ''dense'' and 
   *  v2 is ''sparse''; place sparse result in `this` matrix */
  def outerFill(v1: DenseTensor1, v2: SparseTensor1): Unit = {
    throw new RuntimeException("Row sparse tensor cannot accept dense outer product")    
  }

  def outerFill(v1: DenseTensor1, v2: DenseTensor1): Unit = throw new RuntimeException("Row sparse tensor cannot accept dense outer product")
  
  def clear() = rows.clear()
  
  def setRow(ri: Int, r: Tensor1) = {
    rows.update(ri, r)
  }
  
}


/**
 * Factor to construct [[ColumnSparseTensor2]] objects
 */
object ColumnSparseTensor2 {
  
  /** Initialize sparse column matrix with all zeros */
  def zeros(r: Int, c: Int) = {
    if (r == 2) new ColumnSparseTensor2(Array(SparseTensor1(c), SparseTensor1(c)), r, c)
    else if (r == 3) new ColumnSparseTensor2(Array(SparseTensor1(c), SparseTensor1(c), SparseTensor1(c)), r, c)
    else if (r == 4) new ColumnSparseTensor2(Array(SparseTensor1(c), SparseTensor1(c), SparseTensor1(c), SparseTensor1(c)), r, c)
    else new ColumnSparseTensor2(r,c) 
  }
}

object RowSparseTensor2 {
  def zeros(r: Int, c: Int) = {
    new RowSparseTensor2(new HashMap[Int,Tensor1](), r, c)
  }
}

/**
 * Factory to construct [[DenseTensor2]] objects.
 */
object DenseTensor2 {
  
  /** Construct matrix with all zeros */
  def zeros(r: Int, c: Int) = { new DenseTensor2(r, c) }
  
  /** Initialize matrix with values according to function `fn` */
  def tabulate(r: Int, c: Int)(fn: (Int, Int) => Float) = {
    val m = zeros(r, c)
    var i = 0; while (i < r) {
      var j = 0; while (j < c) {
        m.update(i, j, fn(i, j))
        j += 1
      }
      i += 1
    }
    m
  }
  
  /** Initialize with random values in [0,1] */
  def rand(r: Int, c: Int) = { tabulate(r, c) { case _ => util.Random.nextFloat } }
}

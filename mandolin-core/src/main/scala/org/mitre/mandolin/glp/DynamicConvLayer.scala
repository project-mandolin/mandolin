package org.mitre.mandolin.glp
import org.mitre.mandolin.util.{ DenseTensor2 => DenseMat, DenseTensor1 => DenseVec, Tensor1 => Vec, Tensor2 => Mat }

object DynamicConvLayer {
  def apply(k: Int, width: Int, eDim: Int) = {
    val ones = DenseVec.ones(k*eDim)
    val thFn = { v: DenseVec => v map { v => 1.0 / (1.0 + math.exp(-v)) } }
    val thDeriv = { v: DenseVec => v :* (ones :- v) }
    def cf(x: DenseVec, y: DenseVec) = 0.0
    new DynamicConvLayer(1, k, width, eDim, new LType(LogisticLType,k*eDim), thFn, thDeriv, cf _)
  }
}

/*
 * Implements a convolution layer for 1D timeseries, sequential data
 * k - denotes the max-k operator parameter - output matrix for this feature will have k columns
 * width - denotes the filter width
 * eDim - the number of rows in the input and output matrices (embedding dimension)
 */
class DynamicConvLayer(li: Int, k: Int, width: Int, eDim: Int, lt: LType,
    actFn: DenseVec => DenseVec,
    actFnDeriv: DenseVec => DenseVec,
    costFn: (DenseVec, DenseVec) => Double) extends DenseNonInputLayer(li, k * eDim, lt) with Serializable { 

  val kArgMax = Array.fill(k * eDim)(0)
  // cf. Kalchbrenner et al 2014 ACL
  // implementation top-level plan
  //  - Implement toy/test examples assuming inputs embedded into d-dimensional space already
  //  - Later, allow training of word/embedding vectors within a DCNN
  //  - just do a single feature map in this iteration
  //  - adding multiple feature maps to each layer should be a straightforward generalization
  
  def getActFnDeriv = actFnDeriv(output)
  def setPrevLayer(l: Layer) = { prevLayer_=(Some(l)) }
  def getCost = costFn(getTarget, output)
  
  def forwardWith(in: Vec, w: Mat, b: Vec, training: Boolean) = {
    val wideConv = convolve(w, in, width)
    val (kMax, kaMax) = kmax(wideConv,eDim)
    kMax += b // assumes each b[i] is added to each entry in kMax[i,:]
    val flattened = new DenseVec(kMax.a) // flatten matrix to vector to conform to GLP data flow
    System.arraycopy(kaMax, 0, kArgMax, 0, k * eDim)
    output := actFn(flattened)
  }
  
  def forward(w: Mat, b: Vec, training: Boolean = true) = {  
    val prevIn = prev.getOutput(training)
    forwardWith(prevIn, w, b, training)    
  }
  
  def setTarget(vv: Vec) : Unit = throw new RuntimeException("Convolutional layer has no target")

  def getTarget = throw new RuntimeException("Convolutional layer has no target")
  
  val grad: Mat = DenseMat.zeros(eDim, width)
  // XXX - note that bgrad here has a single bias for each embedding dimension rather than one for each output
  val bgrad: DenseVec = DenseVec.zeros(eDim)
    
  def getGradient(w: Mat, b: Vec) : (Mat, Vec) = {
    backward(w, b)
    getGradient
  }
  
  private def backward(w: Mat, b: Vec) = {
    // update bias gradients
    var i = 0; while (i < dim) {
      val bind = i / k
      bgrad(bind) += delta(i)      
      i += 1
    }    
    // update filter/weight gradients
    val rr = w.getDim1
    val cc = w.getDim2
    val prevIn = prev.getOutput(true)
    val inMatrix = prevIn.toTensor2(eDim)
    val ncols = inMatrix.getDim2    
    var j = 0
    i = 0; while (i < k * eDim) {      
      val dd = delta(i)        
      val convInd = kArgMax(i)
      val rowId = convInd % eDim // this is the input 'row'
      val colId = convInd / eDim // col id in convolution matrix
      j = 0; while (j < width) {
        val inCol = colId - j
        if ((inCol >= 0) && (inCol < ncols)) {
          val inVal = inMatrix(rowId,inCol)
          grad(rowId, j) += dd * inVal
        }        
        j += 1
      }
      i += 1
    }
    prev match {
      // XXX - this really assumes an Embedding Layer as the previous layer
      case p: DenseNonInputLayer =>
        i = 0; while (i < k * eDim) {      
        val dd = delta(i)        
        val convInd = kArgMax(i)
        val rowId = convInd % eDim // this is the input 'row'
        val colId = convInd / eDim // col id in convolution matrix
        j = 0; while (j < width) {
          val inCol = colId - j
          if ((inCol >= 0) && (inCol < ncols)) {
            p.delta(rowId) += dd * w(rowId, j)
          }        
          j += 1
        }
        i += 1
        }    
      case _ =>
    }
  }
  
  def getGradientWith(in: Vec, out: Vec, w: Mat, b: Vec) : (Mat, Vec) = 
    getGradient(w, b)
  
  def getGradient : (Mat, Vec) = (grad, bgrad)
  
  def copy() = {
    val cur = this
    val nl = new DynamicConvLayer(li, k, width, eDim, lt, actFn, actFnDeriv, costFn) {
      override val grad = cur.grad.copy
      override val bgrad = cur.bgrad.copy
    }
    nl
  }

  def sharedWeightCopy() = {
    val cur = this
    val nl = new DynamicConvLayer(li, k, width, eDim, lt, actFn, actFnDeriv, costFn) {
      override val grad = cur.grad.copy
      override val bgrad = cur.bgrad.copy
    }
    nl
  }

  def convolve(filter: Mat, input: Vec, m: Int): Mat = {
    val ninputs = input.getDim / eDim // number of original inputs is input vector / embedding dimension
    val nconvs = ninputs + m - 1
    val noutputs = nconvs * eDim // this assumes a WIDE convolution
    val output = DenseMat.zeros(eDim, nconvs)
    var j = 0; while (j < nconvs) {
      var y = 0; while (y < m) {
        val i = j - m + y + 1
        if ((i >= 0) && (i < ninputs)) { // element-wise product if 'i' not out-of bounds
          var x = 0; while (x < eDim) {
            val inputInd = i * eDim + x // index within original flat input representation
            output(x,j) += (filter(x, y) * input(inputInd))
            x += 1
          }
        }
        y += 1
      }
      j += 1
    }
    output
  }
  
  def shift(mat: Mat, inds: Array[Int], d: Int, nv: Double, ni: Int, curMin: Int) = {
    val end = mat.getDim2 - 1
    val offset = d * mat.getDim2
    for (i <- 0 to end) {      
      if (i > curMin) {
        mat(d,i-1) = mat(d,i)
        inds(offset + i - 1) = inds(offset + i)
      }
    }    
    mat(d,end) = nv
    inds(offset + end) = ni
    var newMin = Double.MaxValue
    var positionMin = -1
    for (i <- 0 to end) {
      if (mat(d,i) < newMin) {
        newMin = mat(d,i)
        positionMin = i
      }
    }    
    (newMin, positionMin)
  }
  
  def kmax(convolution: Mat, dim: Int) : (DenseMat,Array[Int]) = {
    val output = DenseMat.zeros(dim,k)
    val argmaxes = Array.fill(dim*k)(0)
    val numConvs = convolution.getDim2
    var curMin = 0
    var minVal = Double.MaxValue
    println("k = " + k)
    for (d <- 0 until dim) {
      for (i <- 0 until k) {
        val cv = convolution(d,i)
        output(d,i) = cv
        argmaxes(d*k + i) = i
        if (cv < minVal) {
          minVal = cv
          curMin = i
        }
      }
      for (i <- k until numConvs) {
        val cv = convolution(d,i)
        if (cv > minVal) {
          val (nv, na) = shift(output, argmaxes, d, cv, i, curMin)
          minVal = nv
          curMin = na
        }
      }
    }
    (output,argmaxes)
  }
}
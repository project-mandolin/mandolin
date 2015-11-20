package org.mitre.mandolin.glp
import org.mitre.mandolin.util.{ DenseTensor2 => DenseMat, DenseTensor1 => DenseVec, Tensor1 => Vec, Tensor2 => Mat }

/*
 * Implements a convolution layer for 1D timeseries, sequential data
 * k - denotes the max-k operator parameter - output matrix for this feature will have k columns
 * width - denotes the filter width
 * eDim - the number of rows in the input and output matrices (embedding dimension)
 */
class DynamicConvLayer(li: Int, k: Int, width: Int, eDim: Int, lt: LType,
    actFn: DenseVec => DenseVec,
    actFnDeriv: DenseVec => DenseVec,
    costFn: (DenseVec, DenseVec) => Double) extends NonInputLayer(li, k * eDim, lt) with Serializable { 

  // cf. Kalchbrenner et al 2014 ACL
  // implementation top-level plan
  //  - Implement toy/test examples assuming inputs embedded into d-dimensional space already
  //  - Later, allow training of word/embedding vectors within a DCNN
  //  - just do a single feature map in this iteration
  //  - adding multiple feature maps to each layer should be a straightforward generalization
  
  def getActFnDeriv = actFnDeriv(output)
  def setPrevLayer(l: Layer) = { prevLayer_=(Some(l)) }
  def getCost = costFn(getTarget, output)
  
  def forwardWith(in: Vec, w: Mat, b: DenseVec, training: Boolean) = {
    val wideConv = convolve(w, in, width)
    val (kMax, kArgMax) = kmax(wideConv,eDim)
    kMax += b
    val flattened = new DenseVec(kMax.a) // flatten matrix to vector to conform to MLP data flow
    output := actFn(flattened)
  }
  
  def forward(w: Mat, b: DenseVec, training: Boolean = true) = {  
    val prevIn = prev.getOutput(training)
    forwardWith(prevIn, w, b, training)    
  }
  
  def setTarget(v: DenseVec) = throw new RuntimeException("Convolutional layer has no target")
  def getTarget = throw new RuntimeException("Convolutional layer has no target")

  
  val grad: Mat = DenseMat.zeros(eDim, width)
  val bgrad: DenseVec = DenseVec.zeros(eDim)
  
  
  def getGradient(w: Mat, b: DenseVec) : (Mat, DenseVec) = {
    //backward(w: Mat, b: DenseVec)
    getGradient
  }
  
  def getGradientWith(in: Vec, out: DenseVec, w: Mat, b: DenseVec) : (Mat, DenseVec) = 
    getGradient(w, b)
  
  def getGradient : (Mat, DenseVec) = (grad, bgrad)
  
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
            output(x,j) += (filter(x, y) * input(i))
            x += 1
          }
        }
        y += 1
      }
      j += 1
    }
    output
  }
  
  def shift(mat: Mat, inds: Mat, d: Int, nv: Double, ni: Int, curMin: Int) = {
    val end = mat.getDim2 - 1
    for (i <- 0 to end) {      
      if (i > curMin) {
        mat(d,i-1) = mat(d,i)
        inds(d,i-1) = inds(d,i)
      }
    }    
    mat(d,end) = nv
    inds(d,end) = ni
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
  
  def kmax(convolution: Mat, dim: Int) : (DenseMat,DenseMat) = {
    val output = DenseMat.zeros(dim,k)
    val argmaxMatrix = DenseMat.zeros(dim, k)
    val numConvs = convolution.getDim2
    var curMin = 0
    var minVal = Double.MaxValue
    println("k = " + k)
    for (d <- 0 until dim) {
      for (i <- 0 until k) {
        val cv = convolution(d,i)
        output(d,i) = cv
        argmaxMatrix(d,i) = i
        if (cv < minVal) {
          minVal = cv
          curMin = i
        }
      }
      for (i <- k until numConvs) {
        val cv = convolution(d,i)
        if (cv > minVal) {
          val (nv, na) = shift(output, argmaxMatrix, d, cv, i, curMin)
          minVal = nv
          curMin = na
        }
      }
    }
    (output,argmaxMatrix)
  }
}
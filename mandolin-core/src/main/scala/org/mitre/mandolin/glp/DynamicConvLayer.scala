package org.mitre.mandolin.glp
import org.mitre.mandolin.util.{DenseTensor2 => DenseMat, DenseTensor1 => DenseVec}
/*
 * Implements a convolution layer for 1D timeseries, sequential data
 */
class DynamicConvLayer { //(i: Int, curDim: Int, lt: LType) { //extends NonInputLayer(i, curDim, lt) {

  // cf. Kalchbrenner et al 2014 ACL
  // implementation top-level plan
  //  - Implement toy/test examples assuming inputs embedded into d-dimensional space already
  //  - Later, allow training of word/embedding vectors within a DCNN
  
  def convolve(filter: DenseMat, input: DenseVec, d: Int, m: Int) : DenseVec = {
    val ninputs = input.getDim / d // number of original inputs is input vector / embedding dimension
    val nconvs = ninputs + m - 1
    val noutputs = nconvs * d // this assumes a WIDE convolution
    val output = DenseVec.zeros(noutputs)
    var j = 0; while (j < nconvs) {
        var y = 0; while (y < m) {
          val i = j - m + y + 1
          if ((i >= 0) && (i < ninputs)) { // element-wise product if 'i' not out-of bounds
            var x = 0; while (x < d) {                  
              val convIndex = (j * d) + x                       
              output(convIndex) += (filter(x,y) * input(i))
              x += 1
            }          
          }
          y += 1
        }
        j += 1
      }
    output
  }  
}
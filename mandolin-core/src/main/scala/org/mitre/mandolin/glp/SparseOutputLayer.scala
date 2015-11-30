package org.mitre.mandolin.glp

import org.mitre.mandolin.util.{DenseTensor1 => DenseVec, Tensor1 => Vec, SparseTensor1 => SparseVec, Tensor2 => Mat }
/**
 * Enables sparse/sampled output layers for e.g. word2vec
 * @author wellner
 */
abstract class SparseOutputLayer(i: Int, curDim: Int, lt: LType) extends NonInputLayer(i, curDim, lt) with Serializable {
  
  var target: SparseVec = SparseVec(curDim)
  var output: SparseVec = SparseVec(curDim)
  
  def getOutput(tr: Boolean) = output
  
  def setTarget(v: DenseVec) = throw new RuntimeException("Dense vector not allowed as target with sparse output layer")
  def setTarget(v: SparseVec) = target = v  
  def getTarget = target

  //def getActFnDeriv = actFnDeriv(output)
  def setPrevLayer(l: Layer) = { prevLayer_=(Some(l)) }
    
}

abstract class NegSampledSoftMax(i: Int, curDim: Int, numSamples: Int) extends SparseOutputLayer(i, curDim, LType(NegSampledSoftMaxLType, curDim)) {
  val rv = new util.Random
  
  def forwardWith(in: Vec, w:Mat, b: DenseVec, training: Boolean): Unit = {
    val to = SparseVec(curDim)
    var tgInds = target.getNonZeros
    for (i <- 0 until numSamples) {
      tgInds = rv.nextInt(curDim) :: tgInds
    }
    tgInds map {ind => (ind, w.rowDot(ind, in))}
  }
}
package org.mitre.mandolin.glp

import org.mitre.mandolin.util.{DenseTensor1 => DenseVec, Tensor1 => Vec, SparseTensor1 => SparseVec, Tensor2 => Mat, 
  ColumnSparseTensor2 => SparseMat, RowSparseTensor2 => RowSparseMat }
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

  def setPrevLayer(l: Layer) = { prevLayer_=(Some(l)) }
    
}

class NegSampledSoftMaxLayer(i: Int, curDim: Int, prevDim: Int, numSamples: Int) 
extends SparseOutputLayer(i, curDim, LType(NegSampledSoftMaxLType(prevDim, numSamples), curDim)) {
  val rv = new util.Random
  
  def logisticFn(x: Double) = 1.0 / (1.0 + math.exp(x))
  
  def setOutput(t: Vec) = t match {
    case v: SparseVec => output = v
    case _ =>
  }
  
  // squared error over sample
  def getCost = {
    var r = 0.0
    output.forEach((i,v) => r += (v - target(i)) * (v - target(i)))
    r
  }
  
  def setTarget(v: Vec) : Unit = { target := v }
  
  def forwardWith(in: Vec, w:Mat, b: Vec, training: Boolean): Unit = {
    val to = SparseVec(curDim)
    val tgInds = org.mitre.mandolin.util.Sampling.sampleWithoutReplacement(curDim, numSamples, target.getNonZeros)
    output.zeroOut()
    tgInds foreach {ind => output.update(ind, logisticFn(w.rowDot(ind, in))) }    
  }
  
  def forward(w: Mat, b: Vec, training: Boolean = true) = {  
    val prevIn = prev.getOutput(training)
    forwardWith(prevIn, w, b, training)    
  }
  
  def copy() = {
    val cur = this
    val nl = new NegSampledSoftMaxLayer(i, curDim, prevDim, numSamples)
    nl
  }

  def sharedWeightCopy() = copy()

  def getGradient(w: Mat, b: Vec) = getGradientWith(prev.getOutput(true), target, w, b)  
  
  def getGradientWith(in: Vec, target: Vec, w: Mat, b: Vec) = {
    val d = target.copy
    d -= output
    val grad = RowSparseMat.zeros(curDim, prevDim)
    d.forEach({(i,v) =>
      val nr = in * v
      grad.setRow(i, nr)
      })
    (grad, d)
  }
  
  def getGradient: (Mat, Vec) = throw new RuntimeException("Unimplemented")
  
}
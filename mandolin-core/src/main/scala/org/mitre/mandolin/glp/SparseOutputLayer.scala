package org.mitre.mandolin.glp

import org.mitre.mandolin.util.{DenseTensor1 => DenseVec, Tensor1 => Vec, SparseTensor1 => SparseVec, Tensor2 => Mat }
/**
 * Enables sparse/sampled output layers for e.g. word2vec
 * @author wellner
 */
abstract class SparseOutputLayer(i: Int, curDim: Int, lt: LType) extends Layer(i, curDim, lt) with Serializable {
  
  val target: Vec = SparseVec(curDim)
  var output: Vec = SparseVec(curDim)
  
  def setTarget(v: DenseVec) = {  }
  //def getTarget = target

  //def getActFnDeriv = actFnDeriv(output)
  def setPrevLayer(l: Layer) = { prevLayer_=(Some(l)) }
  
  //def forwardWith(in: Vec, w: Mat, b: DenseVec, training: Boolean) = {
}
package org.mitre.mandolin.glp

import org.mitre.mandolin.util.{ DenseTensor2 => DenseMat, DenseTensor1 => DenseVec, 
  ColumnSparseTensor2 => SparseMat, Tensor1 => Vec, Tensor2 => Mat }

/**
 * Represents an embedding from sparse 1-hot inputs to a d-dimensional space
 * Handles sequences of sparse 1-hot representations as an embedding sequence
 * This is useful in a DCNN or for language models or word2vec like setups
 * @param li layer index - should nearly always be the layer just have the sparse input sequence layer
 * @param eDim embedding dimension size
 * @param vocabSize size of the vocabulary
 * @param lt layer type 
 * @author wellner
 */
class SeqEmbeddingLayer(li: Int, eDim: Int, vocabSize: Int, lt: LType) 
extends NonInputLayer(li, eDim*vocabSize, lt) {
 
  def getActFnDeriv = output
  def setPrevLayer(l: Layer) = { prevLayer_=(Some(l)) }
  def getCost = throw new RuntimeException("Cost fn not available for embedding layer")
  
  var sequenceLength = 0
  
  /** Input representation dimension is seqLen*vobabSize with seqLen non-zero values 
   *  Output should be eDim * seqLen laid out column-wise
   *  Weights here would be eDim x vocabSize, representing the embedding
   */
  def forwardWith(in: Vec, w: Mat, b: DenseVec, training: Boolean) = {
    val seqLen = in.getDim / vocabSize
    sequenceLength = seqLen // set this to reuse during backprop step
    output := DenseVec.zeros(eDim * seqLen)
    in.forEach({(i,v) =>
      val vocabInd    = i % vocabSize  // index into vocabulary space
      val seqPosition = i / vocabSize  // item within input sequence
      var j = 0; while (j < eDim) {    // obtain embedding as the output
        val oi = (seqPosition * eDim) + j       // output index which is embedding index + position in sequence
        output(oi) = w(j,vocabInd) * v
        j += 1
      }
    })    
  }
  
  def forward(w: Mat, b: DenseVec, training: Boolean = true) = {  
    val prevIn = prev.getOutput(training)
    forwardWith(prevIn, w, b, training)    
  }
  
  def setTarget(v: DenseVec) = throw new RuntimeException("Convolutional layer has no target")
  def getTarget = throw new RuntimeException("Convolutional layer has no target")
  
  val grad: Mat = SparseMat.zeros(eDim, vocabSize)
  val bgrad: DenseVec = DenseVec.zeros(eDim)
    
  def getGradient(w: Mat, b: DenseVec) : (Mat, DenseVec) = {
    backward(w: Mat, b: DenseVec)
    getGradient
  }
  
  private def backward(w: Mat, b: DenseVec) = {
    // assume this is always the layer just after the input layer, so no need to backprop deltas
    grad.clear() // clear gradient
    val in = prev.getOutput(true)
    in.forEach({(i,v) =>
      val vocabInd    = i % vocabSize  // index into vocabulary space
      val seqPosition = i / vocabSize  // item within input sequence
      var j = 0; while (j < eDim) {   
        val di = (seqPosition * eDim) + j       // delta/error index which is embedding index + position in sequence
        grad(j,vocabInd) += delta(di) * v
        j += 1
      }
    })
    
  }
  
  def getGradientWith(in: Vec, out: DenseVec, w: Mat, b: DenseVec) : (Mat, DenseVec) = 
    getGradient(w, b)
  
  def getGradient : (Mat, DenseVec) = (grad, bgrad)
  
  def copy() = {
    val cur = this
    val nl = new SeqEmbeddingLayer(li, eDim, vocabSize, lt) {
      override val grad = cur.grad.copy
      override val bgrad = cur.bgrad.copy
    }
    nl
  }

  def sharedWeightCopy() = {
    val cur = this
    val nl = new SeqEmbeddingLayer(li, eDim, vocabSize, lt) {
      override val grad = cur.grad.copy
      override val bgrad = cur.bgrad.copy
    }
    nl
  }

}


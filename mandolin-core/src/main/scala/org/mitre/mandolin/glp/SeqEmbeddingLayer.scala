package org.mitre.mandolin.glp

import org.mitre.mandolin.util.{ DenseTensor2 => DenseMat, DenseTensor1 => DenseVec, 
  ColumnSparseTensor2 => SparseMat, Tensor1 => Vec, Tensor2 => Mat }

/**
 * Represents an embedding from sparse 1-hot inputs to a d-dimensional space
 * Handles sequences of sparse 1-hot representations as an embedding sequence
 * This is useful in a DCNN or for language models or word2vec like setups. Note that the effective
 * number of outputs is eDim * seqLen.
 * @param li layer index - should nearly always be the layer just have the sparse input sequence layer
 * @param eDim embedding dimension size
 * @param vocabSize size of the vocabulary
 * @param lt layer type 
 * @author wellner
 */
class SeqEmbeddingLayer(li: Int, eDim: Int, vocabSize: Int, lt: LType, fixedSeqLen: Int = 0) 
extends DenseNonInputLayer(li, eDim, lt) {
 
  
  def getActFnDeriv = output
  def setPrevLayer(l: Layer) = { prevLayer_=(Some(l)) }
  def getCost = throw new RuntimeException("Cost fn not available for embedding layer")
  
  def setTarget(vv: Vec) : Unit = throw new RuntimeException("Embedding layer has no target")
  
  override def getNumberOfOutputs = if (fixedSeqLen > 0) fixedSeqLen * eDim else eDim 
    
  
  var sequenceLength = 0
  
  /** Input representation dimension is seqLen*vobabSize with seqLen non-zero values 
   *  Output should be eDim * seqLen laid out column-wise
   *  Thus:
   *     1 2 3 4 5
   *     6 7 8 9 10 
   *  is the layout with just two embedding dimensions and 5 sequence positions
   *  Weights here would be vocabSize x eDim, representing the embedding
   */
  def forwardWith(in: Vec, w: Mat, b: Vec, training: Boolean) = {
    val seqLen = in.getDim / vocabSize
    sequenceLength = seqLen // set this to reuse during backprop step
    output = DenseVec.zeros(eDim * seqLen)
    delta  = DenseVec.zeros(eDim * seqLen) // set delta at start of forward pass
    val outMat = new DenseMat(output.a, eDim, seqLen)
    in.forEach({(i,v) =>
      val vocabInd    = i % vocabSize  // index into vocabulary space
      val seqPosition = i / vocabSize  // item within input sequence
      var j = 0; while (j < eDim) {    // obtain embedding as the output        
        outMat(j,seqPosition) = w(j, vocabInd) * v
        j += 1
      }
    })    
    println("Finished sequence embedding without output = ")
    println(output)
  }
  
  def forward(w: Mat, b: Vec, training: Boolean = true) = {  
    val prevIn = prev.getOutput(training)
    forwardWith(prevIn, w, b, training)    
  }
  
  def setTarget(v: DenseVec) = throw new RuntimeException("Convolutional layer has no target")
  def getTarget = throw new RuntimeException("Convolutional layer has no target")
  
  val grad: Mat = SparseMat.zeros(eDim, vocabSize)
  val bgrad: DenseVec = DenseVec.zeros(eDim)  // biases not actually used for embedding layer
    
  def getGradient(w: Mat, b: Vec) : (Mat, Vec) = {
    backward(w, b)
    getGradient
  }
  
  private def backward(w: Mat, b: Vec) = {
    // assume this is always the layer just after the input layer, so no need to backprop deltas
    grad.clear() // clear gradient
    val in = prev.getOutput(true)
    val deltaMat = delta.toTensor2(eDim)
    println("Delta = " + deltaMat)
    in.forEach({(i,v) =>
      val vocabInd    = i % vocabSize  // index into vocabulary space
      val seqPosition = i / vocabSize  // item within input sequence
      val offset = seqPosition * eDim
      var j = 0; while (j < eDim) {
        val cv = grad(j,vocabInd)
        grad.update(j,vocabInd, cv + (deltaMat(j, seqPosition) * v))
        println("cv = " + cv + " update = " + (deltaMat(j, seqPosition) * v))
        println("Setting grad " + j + ", " + vocabInd + " to " + grad(j,vocabInd))
        j += 1
      }
    })    
  }
  
  def getGradientWith(in: Vec, out: Vec, w: Mat, b: Vec) : (Mat, Vec) = 
    getGradient(w, b)
  
  def getGradient : (Mat, DenseVec) = (grad, bgrad)
  
  def copy() = {
    val cur = this
    val nl = new SeqEmbeddingLayer(li, eDim, vocabSize, lt, fixedSeqLen) {
      override val grad = cur.grad.copy
      override val bgrad = cur.bgrad.copy
    }
    nl
  }

  def sharedWeightCopy() = {
    val cur = this
    val nl = new SeqEmbeddingLayer(li, eDim, vocabSize, lt, fixedSeqLen) {
      override val grad = cur.grad.copy
      override val bgrad = cur.bgrad.copy
    }
    nl
  }

}


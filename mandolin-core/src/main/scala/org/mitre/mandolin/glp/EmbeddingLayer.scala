package org.mitre.mandolin.glp

import org.mitre.mandolin.util.{ DenseTensor2 => DenseMat, DenseTensor1 => DenseVec, 
  ColumnSparseTensor2 => SparseMat, Tensor1 => Vec, Tensor2 => Mat, StaticSparseTensor1 }

/**
 * @author wellner
 */
class EmbeddingLayer(li: Int, eDim: Int, vocabSize: Int, _ll: LType) 
extends DenseNonInputLayer(li, eDim, _ll) with Serializable {
  
  def getActFnDeriv = output
  def setPrevLayer(l: Layer) = { prevLayer_=(Some(l)) }
  def getCost = throw new RuntimeException("Cost fn not available for embedding layer")
  
  def setTarget(vv: Vec) : Unit = throw new RuntimeException("Embedding layer has no target")
  
  
  def forwardWith(in: Vec, w: Mat, b: Vec, training: Boolean) = {    
    var j = 0; while (j < eDim) {
      output(j) = 0.0f
      j += 1
    }
    var ss = 0.0f
    in match {
      case inV: StaticSparseTensor1 => 
        var i = 0; while (i < inV.len) {
          val ind = inV.indArray(i)
          val vl = inV.valArray(i)
          ss += vl
          var j = 0; while (j < eDim) {
            val cv = output(j)
            output.update(j, cv + w(j,ind) * vl)
            j += 1
          }
          i += 1
        }
      case _ => throw new RuntimeException("Embedding layer requires static sparse input vectors")
    }
    output *= (1.0f / ss)    
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
    grad.clear() // clear gradient
    val in = prev.getOutput(true)
    in.forEach({(i,v) =>
      var j = 0; while (j < eDim) {
        val cv = grad(j,i)
        grad.update(j,i, cv + (delta(j) * v))
        j += 1
      }
    })    
  }
  
  def getGradientWith(in: Vec, out: Vec, w: Mat, b: Vec) : (Mat, Vec) = 
    getGradient(w, b)
  
  def getGradient : (Mat, DenseVec) = (grad, bgrad)
  
  def copy() = {
    val cur = this
    val nl = new EmbeddingLayer(li, eDim, vocabSize, _ll) {
      override val grad = cur.grad.copy
      override val bgrad = cur.bgrad.copy
    }
    nl
  }

  def sharedWeightCopy() = {
    val cur = this
    val nl = new EmbeddingLayer(li, eDim, vocabSize, _ll) {
      override val grad = cur.grad.copy
      override val bgrad = cur.bgrad.copy
    }
    nl
  }
  
  
}
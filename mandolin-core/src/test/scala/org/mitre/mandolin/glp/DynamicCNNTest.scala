package org.mitre.mandolin.glp

import org.scalatest._
import org.mitre.mandolin.util.{DenseTensor1 => DenseVec}

class DynamicCNNTest extends FlatSpec with Matchers {

  import org.mitre.mandolin.glp._
  
  val thFn = { v: DenseVec => v }
  val thDeriv = { v: DenseVec => DenseVec.ones(9) }
    
  "An Embedding NN" should "proceed with FF pass without error" in {
    val l0 = new SparseInputLayer(10, 0.0)
    val l1 = new SeqEmbeddingLayer(1, 3, 10, LType(SeqEmbeddingLType, 3))
    val l2 = new DynamicConvLayer(2, 3, 3, 3, LType(DynamicConvLType, 3), thFn, thDeriv, {(_,_) => 0.0})
    val l2OutDim = 3 * 3  // embedding dimension * k (k-pooling)
    val l3 = WeightLayer.getOutputLayer(new SoftMaxLoss(2), LType(SoftMaxLType, 2) , l2OutDim, 2, false)
    l1.setPrevLayer(l0)
    l2.setPrevLayer(l1)
    l3.setPrevLayer(l2)
    val layers = IndexedSeq(l1,l2,l3)
    val ann = new ANNetwork(l0, layers, true)
    val vecIngest = new SequenceOneHotExtractor(new org.mitre.mandolin.util.IdentityAlphabet(2), 10)
    val fac1 = vecIngest.extractFeatures("0 3 6 8 9 6")
    val glpW = ann.generateRandomWeights
    println("Random weights = ")
    for (i <- 0 until 2) {
      println(glpW.wts.w(2)._1.getRow(i))
    }
    ann.forwardPass(fac1.getInput, fac1.getOutput, glpW, true)    
    println("output = " + ann.outLayer.getOutput(false))
    val revGrads = for (i <- 2 to 0 by -1) yield {
      println("Backprop gradients for layer " + i)
      //println("l2 delta size = " + l2.delta.getDim)
      val (w, b) = glpW.wts.get(i)
      layers(i).getGradient(w, b) // backward pass and return gradient at that layer
    }
    revGrads.reverse
    println("Dynamic CNN test finished")
  }
}
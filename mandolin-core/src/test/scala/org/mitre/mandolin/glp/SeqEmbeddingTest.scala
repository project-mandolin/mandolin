package org.mitre.mandolin.glp

import org.scalatest._
import org.mitre.mandolin.util.{DenseTensor1 => DenseVec}

/**
 * @author wellner
 */
class SeqEmbeddingTest extends FlatSpec with Matchers {
  
  import org.mitre.mandolin.glp._
  
  "An Embedding NN with NegativeSampledSoftmax Output" should "Train" in {
    println("Starting neg softmax test 1")
    // vocab dimension here is 10
    val inSeqLen = 4
    val eDim = 3
    val vocabSize = 10
    val l0 = new SparseInputLayer(vocabSize)
    val l1 = new SeqEmbeddingLayer(1, eDim, vocabSize, LType(SeqEmbeddingLType(inSeqLen), eDim), inSeqLen)
    // should be a pooling layer here to arrive at fixed number of inputs to last layer
    // for this test, assume sequence will have the correct length
    // here assume length is 4 and with embedding dim = 3, this is 12 outputs from embedding layer
    val l2 = new NegSampledSoftMaxLayer(2, vocabSize, eDim * inSeqLen, 5)
    l1.setPrevLayer(l0)
    l2.setPrevLayer(l1)
    val layers = IndexedSeq(l1,l2)
    val ann = new ANNetwork(l0, layers, true)
    println("Seq embedding test...")
    val vecIngest = new SequenceOneHotExtractor(new org.mitre.mandolin.util.IdentityAlphabet(vocabSize), vocabSize)
    val fac1 = vecIngest.extractFeatures("5 1 2 8 9")
    println("About to generate initial weights")
    val glpW = ann.generateRandomWeights
    println("Starting forward pass ... ")
    ann.forwardPass(fac1.getInput, fac1.getOutput, glpW, true)
    println("Finished forward pass...")
    
    val revGrads = for (i <- 1 to 0 by -1) yield {
      //println("l2 delta size = " + l2.delta.getDim)
      val (w, b) = glpW.wts.get(i)
      layers(i).getGradientWith(fac1.getInput, fac1.getOutput, w, b) // backward pass and return gradient at that layer
    }
    
  }
}
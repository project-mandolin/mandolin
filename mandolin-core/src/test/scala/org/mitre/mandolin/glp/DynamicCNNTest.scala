package org.mitre.mandolin.glp

import org.scalatest._

class DynamicCNNTest extends FlatSpec with Matchers {

  import org.mitre.mandolin.glp._
  
  "An Embedding NN" should "proceed with FF pass without error" in {
    val l0 = new SparseInputLayer(10, 0.0)
    val l1 = new SeqEmbeddingLayer(1, 3, 10, LType(SeqEmbeddingLType, 3))
    val l2 = WeightLayer.getOutputLayer(new SoftMaxLoss(2), LType(SoftMaxLType, 2) , 3, 2, false)
    l1.setPrevLayer(l0)
    l2.setPrevLayer(l1)
    val ann = new ANNetwork(l0, IndexedSeq(l1,l2), true)
    val vecIngest = new SequenceOneHotExtractor(new org.mitre.mandolin.util.IdentityAlphabet(2), 10)
    val fac1 = vecIngest.extractFeatures("0 3 6 8 9")
    val glpW = ann.generateRandomWeights
    ann.forwardPass(fac1.getInput, fac1.getOutput, glpW, true)    
    println("Cost = " + ann.getCost)
  }
}
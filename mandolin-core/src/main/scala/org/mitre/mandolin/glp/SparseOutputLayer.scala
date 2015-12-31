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

class NegSampledSoftMaxLayer(i: Int, curDim: Int, prevDim: Int, numSamples: Int, lt: LType) 
extends SparseOutputLayer(i, curDim, lt) {
  
  val unigramTableSize : Int = 10000
  val smoothFactor : Double = 0.75

  
  def getFreqArray(ff: String) : Array[Int] = {
    val ab = new collection.mutable.ArrayBuffer[Int]()
    io.Source.fromFile(ff).getLines foreach {l => ab append l.toInt }
    ab.toArray
  }
  
  def computeFrequencyTable(freqFile: String) : Array[Int] = {
    val ut = Array.fill(unigramTableSize)(0)
    val ft = getFreqArray(freqFile)
    val ftSize = ft.length
    var i = 0
    var total = 0.0
    var a = 0; while (a < ftSize) { total += math.pow(ft(a),smoothFactor) ;a += 1}
    var d1 = math.pow(ft(0), smoothFactor) / total
    a = 0; while (a < unigramTableSize) {
      ut(a) = i
      if ((a.toDouble / unigramTableSize) > d1) {
        i += 1
        if (i < ftSize) d1 += math.pow(ft(i),smoothFactor) / total 
      }
      if (i >= ftSize) {
        i = ftSize - 1
      }
      a += 1
    }
    ut
  } 

  val rv = new util.Random
  val freqTable = {
    lt.designate match {
      case NegSampledSoftMaxLType(cd,pd,ff) =>
        computeFrequencyTable(ff)
      case _ => throw new RuntimeException("Frequency table expected for negative sampling softmax!!")
    }
  }
  
  def logisticFn(x: Float) = 1.0f / (1.0f + math.exp(-x)).toFloat
  
  
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
    val tgInds = org.mitre.mandolin.util.Sampling.sampleWithoutReplacement(freqTable, numSamples, target.getNonZeros)
    val ov = SparseVec.getOnes(curDim,tgInds.toArray)
    /*
    var ss = 0.0
    ov.mapInPlace{(i,v) =>
      val oo = math.exp(w.rowDot(i,in)) // no bias here b(i)
      ss += oo
      oo}
      *
      * ov.mapInPlace{(i,v) => v / ss} 
      */
    ov.mapInPlace{(i,v) => logisticFn(w.rowDot(i,in))} // not full soft-max, logistic outputs
    output := ov
  }
  
  def forward(w: Mat, b: Vec, training: Boolean = true) = {  
    val prevIn = prev.getOutput(training)
    forwardWith(prevIn, w, b, training)    
  }
  
  def copy() = {
    val cur = this
    val nl = new NegSampledSoftMaxLayer(i, curDim, prevDim, numSamples, lt)
    nl
  }

  def sharedWeightCopy() = copy()

  def getGradient(w: Mat, b: Vec) = getGradientWith(prev.getOutput(true), target, w, b)  
  
  def getGradientWith(in: Vec, target: Vec, w: Mat, b: Vec) = {
    val d = target.copy // output.copy
    d -= output // target
    prev match {
      case p: DenseNonInputLayer =>
        var i = 0; while (i < p.delta.getDim) {
          p.delta(i) = 0.0f
          d.forEach{(j,v) =>
            val cv = p.delta(i)
            p.delta.update(i, cv + w(j,i) * v)
            }
          i += 1
        }
      case _ =>
    }    
    val grad = RowSparseMat.zeros(curDim, prevDim)
    d.forEach({(i,v) => grad.setRow(i, in * v) })
    // return an empty vec here for bias .. i.e. no bias
    (grad, SparseVec(curDim))
  }
  
  def getGradient: (Mat, Vec) = throw new RuntimeException("Unimplemented")
  
}
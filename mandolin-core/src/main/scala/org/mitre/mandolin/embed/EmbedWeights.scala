package org.mitre.mandolin.embed

import org.mitre.mandolin.optimize.{Weights, LossGradient, Updater}
import org.mitre.mandolin.util.{Alphabet, Tensor1 => Vec, Tensor2 => Mat, DenseTensor2 => DenseMat, DenseTensor1 => DenseVec}

/*
 * Note that these weights are represented using the transpose of GLP weights.
 * That is, dim(W) = (l-1, l) has dim(l-1) rows and dim(l) columns where dim(l-1) is the number of inputs and dim(l)
 * is the number of outputs/dimensions of current layer
 */
class EmbedWeights(val embW: Mat, val outW: Mat, m: Float) extends Weights[EmbedWeights](m) with Serializable {

  val numWeights = embW.getSize + outW.getSize
  def compress(): Unit = {}
  def decompress(): Unit = {}
  def weightAt(i: Int) = throw new RuntimeException("Not implemented")
  
  def compose(otherWeights: EmbedWeights) = {
    this *= mass
    otherWeights *= otherWeights.mass
    this ++ otherWeights
    val nmass = mass + otherWeights.mass
    this *= (1.0f / nmass)
    new EmbedWeights(this.embW, this.outW, nmass)
  }
  
  def add(otherWeights: EmbedWeights): EmbedWeights = {
    this += otherWeights
    this
  }

  def addEquals(otherWeights: EmbedWeights): Unit = {
    embW += otherWeights.embW
    outW += otherWeights.outW
  }

  def timesEquals(v: Float) = { 
    embW *= v
    outW *= v
    }
  
  def updateFromArray(ar: Array[Float]) = { throw new RuntimeException("array update for NN not implemented") }
  def updateFromArray(ar: Array[Double]) = { throw new RuntimeException("array update for NN not implemented") }
  def asArray = { throw new RuntimeException("array update for NN not implemented") }

  def l2norm = throw new RuntimeException("Norm not implemented")
  def copy() = new EmbedWeights(embW.copy(), outW.copy(), m)
  def asTensor1() = throw new RuntimeException("Tensor construction not available with deep weight array")
  def sharedWeightCopy() = new EmbedWeights(embW, outW, m)
  
  def exportWithMapping(mapping: Alphabet, outFile: java.io.File) = {
    val os = new java.io.BufferedWriter(new java.io.FileWriter(outFile))
    val inv = mapping.getInverseMapping
    os.write(embW.getDim1.toString)
    os.write(' ')
    os.write(embW.getDim2.toString)
    os.write('\n')
    var i = 0; while (i < embW.getDim1) {
      os.write(inv(i))
      var j = 0; while (j < embW.getDim2) {
        os.write(' ')
        os.write(embW(i,j).toString)
        j += 1
      }
      os.write('\n')
      i += 1
    }
    os.close
  }  
}

class EmbedGradient extends LossGradient[EmbedGradient](0.0) {
  def add(o: EmbedGradient) = this
  def asArray = throw new RuntimeException("Norm not implemented")
}

class NullUpdater extends Updater[EmbedWeights, EmbedGradient, NullUpdater] with Serializable {
  
  // trick here is to use updater to keep track of total instances processed
  // actual parameter updates happen within evaluator for efficiency
  @volatile var totalProcessed = 0
  
  // these are null ops
  def updateWeights(g: EmbedGradient, w: EmbedWeights): Unit = {}
  def resetLearningRates(v: Float): Unit = {}
  def copy() : NullUpdater = new NullUpdater
  def compose(u: NullUpdater) : NullUpdater = this
}

class EmbedAdaGradUpdater(initialLearningRate: Float, val embSqG: Array[Float], val outSqG: Array[Float]) 
extends Updater[EmbedWeights, EmbedGradient, EmbedAdaGradUpdater] with Serializable {
  def updateWeights(g: EmbedGradient, w: EmbedWeights): Unit = {}
  def resetLearningRates(v: Float): Unit = {}
  def copy() : EmbedAdaGradUpdater = new EmbedAdaGradUpdater(initialLearningRate,embSqG, outSqG)
  def compose(u: EmbedAdaGradUpdater) : EmbedAdaGradUpdater = this
  
  @inline
  final private def fastSqrt(x: Float) : Float = 
    java.lang.Float.intBitsToFloat(532483686 + (java.lang.Float.floatToRawIntBits(x) >> 1))
    
  def updateEmbeddingSqG(i: Int, wArr: Array[Float], g: Float) = {    
    embSqG(i) += g * g
    wArr(i) += (initialLearningRate * g / (initialLearningRate + fastSqrt(embSqG(i))))
  }
  
  def updateOutputSqG(i: Int, wArr: Array[Float], g: Float) = {
    outSqG(i) += g * g
    wArr(i) += (initialLearningRate * g / (initialLearningRate + fastSqrt(outSqG(i))))
  }
  
}

object EmbedWeights {
  
  def apply(eDim: Int, vDim: Int) = {
     val embW = DenseMat.zeros(vDim, eDim)
     val outW = DenseMat.zeros(vDim, eDim)
     var i = 0; while (i < vDim) {
       var j = 0; while (j < eDim) {
         val nv = (util.Random.nextFloat - 0.5f) / eDim
         embW.update(i, j, nv)
         j += 1
       }
       i += 1
     }
     new EmbedWeights(embW, outW, 1.0f)
  }
}
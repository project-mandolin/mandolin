package org.mitre.mandolin.embed

import org.mitre.mandolin.optimize.{Weights, LossGradient, Updater}
import org.mitre.mandolin.util.{Alphabet, Simple2DArray, Tensor1 => Vec, Tensor2 => Mat, DenseTensor2 => DenseMat, DenseTensor1 => DenseVec}

import xerial.larray._

/*
 * Note that these weights are represented using the transpose of MMLP weights.
 * That is, dim(W) = (l-1, l) has dim(l-1) rows and dim(l) columns where dim(l-1) is the number of inputs and dim(l)
 * is the number of outputs/dimensions of current layer
 */
class EmbedWeights(val embW: Simple2DArray, val outW: Simple2DArray, m: Float) extends Weights[EmbedWeights](m) with Serializable {

  val numWeights = embW.getSize + outW.getSize
  def compress(): Unit = {}
  def decompress(): Unit = {}
  def weightAt(i: Int) = throw new RuntimeException("Not implemented")
  
  var fastEmbW : Option[LArray2D[Float]] = None
  var fastOutW : Option[LArray2D[Float]] = None
  var accelerated = false
  def accelerate = {
    if (!accelerated) {
      val emb = LArray2D.of[Float](embW.getDim2, embW.getDim1)
      val out = LArray2D.of[Float](outW.getDim2, outW.getDim1)
      var i = 0; while (i < embW.getDim1) {
        var j = 0; while (j < embW.getDim2) {
          emb(i,j) = embW(i,j)
          out(i,j) = outW(i,j)
          j += 1
        }
        i += 1
      }
      fastEmbW_=(Some(emb))
      fastOutW_=(Some(out))
      accelerated_=(true)
    }
  }
  
  def decelerate = {
    if (accelerated) {
      val emb = fastEmbW.get
      val out = fastOutW.get
      var i = 0; while (i < embW.getDim1) {
        var j = 0; while (j < embW.getDim2) {
          embW(i,j) = emb(i,j)
          outW(i,j) = out(i,j)
          j += 1
        }
        i += 1
      }
    accelerated_=(false)
    fastEmbW_=(None)
    fastOutW_=(None)
    }    
  }
  
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

abstract class EmbedUpdater extends Updater[EmbedWeights, EmbedGradient, EmbedUpdater] with Serializable {
  def updateEmbeddingSqG(i: Int, j: Int, wArr: Simple2DArray, g: Float) : Unit
  
  def updateEmbeddingSqG(i: Int, j: Int, wArr: LArray2D[Float], g: Float) : Unit
  
  def updateOutputSqG(i: Int, j: Int, wArr: Simple2DArray, g: Float) : Unit
  
  def updateOutputSqG(i: Int, j: Int, wArr: LArray2D[Float], g: Float) : Unit
  
  def updateNumProcessed() : Unit
  
  def getArraySeq : Seq[Array[Float]]
  
  def getTotalProcessed : Int
  def getCurrentRate : Float
  
}


class NullUpdater(val initialLearnRate: Float, val decay: Float) extends EmbedUpdater with Serializable {
  
  @volatile var totalProcessed = 0.0f
  var currentRate = 0.0f
  
  def getZeroUpdater = new NullUpdater(initialLearnRate, decay)
  
  def getTotalProcessed : Int = totalProcessed.toInt
  def getCurrentRate = currentRate
  
  def updateFromArray(ar: Array[Float]) = {}
  
  // these are null ops
  def updateWeights(g: EmbedGradient, w: EmbedWeights): Unit = {}
  def resetLearningRates(v: Float): Unit = {
    this.totalProcessed = v    
  }
  def copy() : NullUpdater = new NullUpdater(initialLearnRate, decay)
  def compose(u: EmbedUpdater) : EmbedUpdater = {    
    this
  }
  def asArray = Array[Float](initialLearnRate)
  
  def compress() = this
  def decompress() = this
  
  def getArraySeq = Seq(Array[Float]())
  
  @inline
  final def updateEmbeddingSqG(i: Int, j : Int, wArr: Simple2DArray, g: Float) = { wArr(i,j) += g * currentRate }
  
  @inline
  final def updateEmbeddingSqG(i: Int, j: Int, wArr: LArray2D[Float], g: Float) : Unit = {
    wArr(i,j) += g * currentRate
  }
  
  @inline
  final def updateOutputSqG(i: Int, j: Int, wArr: Simple2DArray, g: Float) = { wArr(i,j) += g * currentRate }

  @inline
  final def updateOutputSqG(i: Int, j: Int, wArr: LArray2D[Float], g: Float) = { wArr(i,j) += g * currentRate }

  @inline
  final def updateNumProcessed() = {
    totalProcessed += 1.0f
    currentRate = initialLearnRate / (1.0f + totalProcessed * decay)
  }

}


class EmbedAdaGradUpdater(val initialLearningRate: Float, val embSqG: Simple2DArray, val outSqG: Simple2DArray) 
extends EmbedUpdater with Serializable {
  
  def this(ir: Float, rows: Int, cols: Int) = this(ir, Simple2DArray.floatArray(rows, cols), Simple2DArray.floatArray(rows,cols))
  
  import org.mitre.mandolin.util.SimpleFloatCompressor
  
  def getTotalProcessed = 0
  def getCurrentRate = initialLearningRate
  
  // val fullSize = embSqG.length
  def getZeroUpdater = 
    new EmbedAdaGradUpdater(initialLearningRate, Simple2DArray.floatArray(embSqG.getDim1, embSqG.getDim2), Simple2DArray.floatArray(outSqG.getDim1, outSqG.getDim2))

  var compressedEmbSqG = (Simple2DArray.byteArray(0, 0), Array.fill(0)(0.0f))
  var compressedOutSqG = (Simple2DArray.byteArray(0, 0), Array.fill(0)(0.0f))
  
  def getArraySeq : Seq[Array[Float]] = embSqG.ars.toSeq ++ outSqG.ars.toSeq
  
  def compress() = {
    if (!isCompressed) {
      val ne = new EmbedAdaGradUpdater(initialLearningRate, Simple2DArray.floatArray(0,0), Simple2DArray.floatArray(0,0))
      ne.compressedEmbSqG_=(SimpleFloatCompressor.compress(embSqG))
      ne.compressedOutSqG_=(SimpleFloatCompressor.compress(outSqG))
      ne.isCompressed_=(true)    
      ne 
    } else this
  }
  
  def decompress() = {
    if (isCompressed) {
      val (eAr,eVls) = compressedEmbSqG
      val (oAr,oVls) = compressedOutSqG
      val nEmb = Array.tabulate(eAr.getDim1){i => Array.tabulate(eAr.getDim2){ j => eVls(eAr(i,j).toInt + 128)}}
      val nOut = Array.tabulate(oAr.getDim1){i => Array.tabulate(oAr.getDim2){ j => oVls(oAr(i,j).toInt + 128)}}
      new EmbedAdaGradUpdater(initialLearningRate, new Simple2DArray(nEmb, eAr.getDim1, eAr.getDim2), new Simple2DArray(nOut, oAr.getDim1, oAr.getDim2))
    } else this
  }
  
  def updateWeights(g: EmbedGradient, w: EmbedWeights): Unit = {}
  def resetLearningRates(v: Float): Unit = {
    /*
    var i = 0; while (i < embSqG.getDim1) {
      var j = 0; while (j < embSqG.getDim2) {
        embSqG(i,j) = v
        outSqG(i,j) = v
        j += 1
      }
      i += 1
    }
    */
  }
  
  def updateFromArray(ar: Array[Float]) : Unit = {
    throw new RuntimeException("Update from array not available")
  }
  
  def asArray = {
    throw new RuntimeException("As array not available")
  }
  
  def copy() : EmbedAdaGradUpdater = {
    val cc = new EmbedAdaGradUpdater(initialLearningRate,embSqG, outSqG)
    // important state
    cc.compressedEmbSqG_=(this.compressedEmbSqG)
    cc.compressedOutSqG_=(this.compressedEmbSqG)
    cc.isCompressed_=(this.isCompressed)
    cc
  }
  
  def compose(_u: EmbedUpdater) : EmbedUpdater = {
    val u = _u match {case x: EmbedAdaGradUpdater => x case _ => throw new RuntimeException("Compose with AdaGrad updater requires AdaGrad updater")}
    if (isCompressed) {
      val (eAr, eVls) = compressedEmbSqG
      val (oAr, oVls) = compressedOutSqG
      val (u_eAr, u_eVls) = u.compressedEmbSqG
      val (u_oAr, u_oVls) = u.compressedOutSqG
      var i = 0; while (i < eAr.getDim1) {
        var j = 0; while (j < eAr.getDim2) {
          eAr(i,j) = (math.max(eAr(i,j), u_eAr(i,j)).toByte)
          oAr(i,j) = (math.max(oAr(i,j), u_oAr(i,j)).toByte)
          j += 1 
        }
        i += 1
      }
      i = 0; while (i < oVls.length) {
        eVls(i) = math.max(eVls(i), u_eVls(i))
        oVls(i) = math.max(oVls(i), u_oVls(i))
        i += 1
      }
      this
    } else {
      var i = 0; while (i < embSqG.getDim1) {
        var j = 0; while (j < embSqG.getDim2) {
          embSqG(i,j) = math.max(embSqG(i,j),u.embSqG(i,j))
          outSqG(i,j) = math.max(outSqG(i,j),u.outSqG(i,j))
          j += 1
        }
        i += 1
      }
      this
    }
  }
  
  @inline
  final private def fastSqrt(x: Float) : Float = 
    java.lang.Float.intBitsToFloat(532483686 + (java.lang.Float.floatToRawIntBits(x) >> 1))
  
  @inline
  final def updateEmbeddingSqG(i: Int, j: Int, wArr: Simple2DArray, g: Float) = {    
    embSqG(i,j) += g * g
    wArr(i,j) += (initialLearningRate * g / (initialLearningRate + fastSqrt(embSqG(i,j))))
  }

  @inline
  final def updateEmbeddingSqG(i: Int, j: Int, wArr: LArray2D[Float], g: Float) = {    
    embSqG(i,j) += g * g
    wArr(i,j) += (initialLearningRate * g / (initialLearningRate + fastSqrt(embSqG(i,j))))
  }

  @inline
  final def updateOutputSqG(i: Int, j: Int, wArr: LArray2D[Float], g: Float) = {
    outSqG(i,j) += g * g
    wArr(i,j) += (initialLearningRate * g / (initialLearningRate + fastSqrt(outSqG(i,j))))
  }
  
  final def updateOutputSqG(i: Int, j: Int, wArr: Simple2DArray, g: Float) = {
    outSqG(i,j) += g * g
    wArr(i,j) += (initialLearningRate * g / (initialLearningRate + fastSqrt(outSqG(i,j))))
  }  
  def updateNumProcessed() = {}
}

object EmbedWeights {
  
  def apply(eDim: Int, vDim: Int) = {
     // val embW = DenseMat.zeros(vDim, eDim)
     // val outW = DenseMat.zeros(vDim, eDim)
    val embW = Simple2DArray.floatArray(vDim, eDim)
    val outW = Simple2DArray.floatArray(vDim, eDim)
     var i = 0; while (i < vDim) {
       var j = 0; while (j < eDim) {
         val nv = (scala.util.Random.nextFloat - 0.5f) / eDim
         embW.update(i, j, nv)
         j += 1
       }
       i += 1
     }
     new EmbedWeights(embW, outW, 1.0f)
  }
}
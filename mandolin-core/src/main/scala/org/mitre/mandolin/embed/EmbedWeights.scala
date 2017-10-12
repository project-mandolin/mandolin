package org.mitre.mandolin.embed

import org.mitre.mandolin.optimize.{Weights, LossGradient, Updater}
import org.mitre.mandolin.util.{Alphabet, Tensor1 => Vec, Tensor2 => Mat, DenseTensor2 => DenseMat, DenseTensor1 => DenseVec}
import xerial.larray._
import xerial.larray._
/*
 * Note that these weights are represented using the transpose of MMLP weights.
 * That is, dim(W) = (l-1, l) has dim(l-1) rows and dim(l) columns where dim(l-1) is the number of inputs and dim(l)
 * is the number of outputs/dimensions of current layer
 */

class EmbedWeights(val embW: LArray2D[Float], val outW: LArray2D[Float], m: Float) extends Weights[EmbedWeights](m) with Serializable {

  val numWeightsLong = embW.rawArray.length + outW.rawArray.length
  val numWeights = numWeightsLong.toInt
  val numEmbWeights = embW.rawArray.length 
  
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
    val rawE = embW.rawArray
    val otherRawE = otherWeights.embW.rawArray
    val rawO = outW.rawArray
    val otherRawO = otherWeights.outW.rawArray    
    var i = 0; while (i < numEmbWeights) {
      rawE(i) += otherRawE(i)
      rawO(i) += otherRawO(i)
      i += 1
    }
  }

  def timesEquals(v: Float) = {
    val rawE = embW.rawArray
    val rawO = outW.rawArray
    var i = 0; while (i < numEmbWeights) {
      rawE(i) *= v
      rawO(i) *= v
    }
  }
  
  def updateFromArray(ar: Array[Float]) = {
    throw new RuntimeException("Update from array not implemented yet ...")    
  }
  
  def updateFromArray(ar: Array[Double]) = { throw new RuntimeException("array update for NN not implemented") }
  def asArray = { throw new RuntimeException("Asrray not implemented yet ...") }

  def l2norm = throw new RuntimeException("Norm not implemented")
  def copy() = {
    val newEW = LArray2D.of[Float](embW.rowSize,embW.colSize)
    val newOW = LArray2D.of[Float](outW.rowSize,outW.colSize)
    embW.rawArray.copyToArray(newEW.rawArray,0L,newEW.rawArray.length)
    outW.rawArray.copyToArray(newOW.rawArray,0L,newOW.rawArray.length)
    new EmbedWeights(newEW, newOW, m)
  }
  def asTensor1() = throw new RuntimeException("Tensor construction not available with deep weight array")
  def sharedWeightCopy() = new EmbedWeights(embW, outW, m)
  
  def exportWithMapping(mapping: Alphabet, outFile: java.io.File) = {
    val os = new java.io.BufferedWriter(new java.io.FileWriter(outFile))
    val inv = mapping.getInverseMapping
    os.write(embW.colSize.toString)
    os.write(' ')
    os.write(embW.rowSize.toString)
    os.write('\n')
    var i = 0; while (i < embW.colSize) {
      os.write(inv(i))
      var j = 0; while (j < embW.rowSize) {
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

abstract class EmbedUpdater[T <: Updater[EmbedWeights, EmbedGradient, T]] extends Updater[EmbedWeights, EmbedGradient, T] with Serializable {
  def updateEmbeddingSqG(i: Long, j: Long, wts: LArray2D[Float], g: Float)
  
  def updateOutputSqG(i: Long, j: Long, wts: LArray2D[Float], g: Float)
  
  def updateNumProcessed() : Unit
  
}

class NullUpdater(val initialLearnRate: Float, val decay: Float) extends EmbedUpdater[NullUpdater] with Serializable {
  
  @volatile var totalProcessed = 0.0f
  var currentRate = 0.0f
  
  // def getZeroUpdater = new NullUpdater(initialLearnRate, decay)
  
  def updateFromArray(ar: Array[Float]) = {}
  
  // these are null ops
  def updateWeights(g: EmbedGradient, w: EmbedWeights): Unit = {}
  def resetLearningRates(v: Float): Unit = {
    this.totalProcessed = v    
  }
  def copy() : NullUpdater = new NullUpdater(initialLearnRate, decay)
  def compose(u: NullUpdater) : NullUpdater = {
    this.totalProcessed += u.totalProcessed    
    this
  }
  def asArray = Array[Float](initialLearnRate)
  
  def compress() = this
  def decompress() = this
  
  @inline
  final def updateEmbeddingSqG(i: Long, j: Long, wts: LArray2D[Float], g: Float) = { wts(i,j) += g * currentRate }
  
  @inline
  final def updateOutputSqG(i: Long, j: Long, wts: LArray2D[Float], g: Float) = { wts(i,j) += g * currentRate }
  
  @inline
  final def updateNumProcessed() = {
    totalProcessed += 1.0f
    currentRate = initialLearnRate / (1.0f + totalProcessed * decay)
  }
}

class EmbedAdaGradUpdater(initialLearningRate: Float, val embSqG: LArray[Float], val outSqG: LArray[Float]) 
extends EmbedUpdater[EmbedAdaGradUpdater] with Serializable {
  
  import org.mitre.mandolin.util.SimpleFloatCompressor
  
  val fullSize = embSqG.length
  // def getZeroUpdater = new EmbedAdaGradUpdater(initialLearningRate, LArray.of[Float](fullSize), LArray.of[Float](fullSize))

  var compressedEmbSqG = (LArray[Byte](), Array[Float]())
  var compressedOutSqG = (LArray[Byte](), Array[Float]())
  
  def compress() = {
    if (!isCompressed) {
      val ne = new EmbedAdaGradUpdater(initialLearningRate, LArray[Float](), LArray[Float]())
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
      //System.err.println("")
      val nEmb = LArray.of[Float](eAr.length)
      val nOut = LArray.of[Float](oAr.length)
      var i = 0; while (i < eAr.length) {
        nEmb(i) = eVls(eAr(i).toInt + 128)
        nOut(i) = oVls(oAr(i).toInt + 128)
        i += 1
      }
      //val nEmb = LArray.tabulate(eAr.length){i => eVls(eAr(i).toInt + 128)}
      //val nOut = LArray.tabulate(oAr.length){i => oVls(oAr(i).toInt + 128)}
      new EmbedAdaGradUpdater(initialLearningRate, nEmb, nOut)
    } else this
  }
  
  def updateWeights(g: EmbedGradient, w: EmbedWeights): Unit = {}
  def resetLearningRates(v: Float): Unit = {
    var i = 0; while (i < fullSize) {
      embSqG(i) = v
      outSqG(i) = v
      i += 1
    }
  }
  
  def updateFromArray(ar: Array[Float]) : Unit = {
    var i = 0; while (i < ar.length) {
      if (i < fullSize)
        embSqG(i) = ar(i)
      else 
        outSqG(i - fullSize) = ar(i)
      i += 1
    }
  }
  
  def asArray = { throw new RuntimeException("As array unavailable as this could exceed standard array limits")
    // Array.tabulate(fullSize*2){i => if (i < fullSize) embSqG(i) else outSqG(i - fullSize) }
  }
  
  def copy() : EmbedAdaGradUpdater = {
    val cc = new EmbedAdaGradUpdater(initialLearningRate,embSqG, outSqG)
    // important state
    cc.compressedEmbSqG_=(this.compressedEmbSqG)
    cc.compressedOutSqG_=(this.compressedEmbSqG)
    cc.isCompressed_=(this.isCompressed)
    cc
  }
  
  def compose(u: EmbedAdaGradUpdater) : EmbedAdaGradUpdater = {
    if (isCompressed) {
      val (eAr, eVls) = compressedEmbSqG
      val (oAr, oVls) = compressedOutSqG
      val (u_eAr, u_eVls) = u.compressedEmbSqG
      val (u_oAr, u_oVls) = u.compressedOutSqG
      var i = 0; while (i < eAr.length) {
        eAr(i) = math.max(eAr(i), u_eAr(i)).toByte
        oAr(i) = math.max(oAr(i), u_oAr(i)).toByte
        i += 1
      }
      i = 0; while (i < oVls.length) {
        eVls(i) = math.max(eVls(i), u_eVls(i))
        oVls(i) = math.max(oVls(i), u_oVls(i))
        i += 1
      }
      this
    } else {
      var i = 0; while (i < fullSize) {
        embSqG(i) = math.max(embSqG(i),u.embSqG(i))
        outSqG(i) = math.max(outSqG(i),u.outSqG(i))
        i += 1
      }
      this
    }
  }
  
  @inline
  final private def fastSqrt(x: Float) : Float = 
    java.lang.Float.intBitsToFloat(532483686 + (java.lang.Float.floatToRawIntBits(x) >> 1))
    
  final def updateEmbeddingSqG(i: Long, j: Long, wts: LArray2D[Float], g: Float) = {
    val offset = i * wts.rowSize + j
    embSqG(offset) += g * g
    wts(i,j) += (initialLearningRate * g / (initialLearningRate + fastSqrt(embSqG(offset))))
  }
  
  final def updateOutputSqG(i: Long, j: Long, wts: LArray2D[Float], g: Float) = {
    val offset = i * wts.rowSize + j
    outSqG(offset) += g * g
    wts(i,j) += (initialLearningRate * g / (initialLearningRate + fastSqrt(outSqG(offset))))
  }  
  def updateNumProcessed() = {}
}

object EmbedWeights {
  
  def apply(eDim: Int, vDim: Int) = {
     val embW = LArray2D.of[Float](eDim, vDim) // recall eDim here is size of each row - i.e. NUMBER of columns
     val outW = LArray2D.of[Float](eDim, vDim)
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

/*
 * class EmbedWeights(val embW: Mat, val outW: Mat, m: Float) extends Weights[EmbedWeights](m) with Serializable {

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
  
  def updateFromArray(ar: Array[Float]) = {
    val eArr = embW.asArray
    val oArr = outW.asArray
    val ss = embW.getSize
    var i = 0; while (i < ar.length) {
      if (i < ss) {
        eArr(i) = ar(i)        
      } else {
        oArr(i - embW.getSize) = ar(i)
      }
      i += 1
    }
    this
  }
  
  def updateFromArray(ar: Array[Double]) = { throw new RuntimeException("array update for NN not implemented") }
  def asArray = {
    val eArr = embW.asArray
    val oArr = outW.asArray
    val ss = embW.getSize    
    Array.tabulate(numWeights){i => if (i < ss) eArr(i) else oArr(i - ss) }
  }

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

abstract class EmbedUpdater[T <: Updater[EmbedWeights, EmbedGradient, T]] extends Updater[EmbedWeights, EmbedGradient, T] with Serializable {
  def updateEmbeddingSqG(i: Int, wArr: Array[Float], g: Float)
  
  def updateOutputSqG(i: Int, wArr: Array[Float], g: Float)
  
  def updateNumProcessed() : Unit
  
}

class NullUpdater(val initialLearnRate: Float, val decay: Float) extends EmbedUpdater[NullUpdater] with Serializable {
  
  @volatile var totalProcessed = 0.0f
  var currentRate = 0.0f
  
  def getZeroUpdater = new NullUpdater(initialLearnRate, decay)
  
  def updateFromArray(ar: Array[Float]) = {}
  
  // these are null ops
  def updateWeights(g: EmbedGradient, w: EmbedWeights): Unit = {}
  def resetLearningRates(v: Float): Unit = {
    this.totalProcessed = v    
  }
  def copy() : NullUpdater = new NullUpdater(initialLearnRate, decay)
  def compose(u: NullUpdater) : NullUpdater = {
    this.totalProcessed += u.totalProcessed    
    this
  }
  def asArray = Array[Float](initialLearnRate)
  
  def compress() = this
  def decompress() = this
  
  @inline
  final def updateEmbeddingSqG(i: Int, wArr: Array[Float], g: Float) = { wArr(i) += g * currentRate }
  
  @inline
  final def updateOutputSqG(i: Int, wArr: Array[Float], g: Float) = { wArr(i) += g * currentRate }
  
  @inline
  final def updateNumProcessed() = {
    totalProcessed += 1.0f
    currentRate = initialLearnRate / (1.0f + totalProcessed * decay)
  }
}

class EmbedAdaGradUpdater(initialLearningRate: Float, val embSqG: Array[Float], val outSqG: Array[Float]) 
extends EmbedUpdater[EmbedAdaGradUpdater] with Serializable {
  
  import org.mitre.mandolin.util.SimpleFloatCompressor
  
  val fullSize = embSqG.length
  def getZeroUpdater = new EmbedAdaGradUpdater(initialLearningRate, Array.fill(fullSize)(0.0f), Array.fill(fullSize)(0.0f))

  var compressedEmbSqG = (Array[Byte](), Array[Float]())
  var compressedOutSqG = (Array[Byte](), Array[Float]())
  
  def compress() = {
    if (!isCompressed) {
      val ne = new EmbedAdaGradUpdater(initialLearningRate, Array(), Array())
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
      val nEmb = Array.tabulate(eAr.length){i => eVls(eAr(i).toInt + 128)}
      val nOut = Array.tabulate(oAr.length){i => oVls(oAr(i).toInt + 128)}
      new EmbedAdaGradUpdater(initialLearningRate, nEmb, nOut)
    } else this
  }
  
  def updateWeights(g: EmbedGradient, w: EmbedWeights): Unit = {}
  def resetLearningRates(v: Float): Unit = {
    var i = 0; while (i < fullSize) {
      embSqG(i) = v
      outSqG(i) = v
      i += 1
    }
  }
  
  def updateFromArray(ar: Array[Float]) : Unit = {
    var i = 0; while (i < ar.length) {
      if (i < fullSize)
        embSqG(i) = ar(i)
      else 
        outSqG(i - fullSize) = ar(i)
      i += 1
    }
  }
  
  def asArray = {
    Array.tabulate(fullSize*2){i => if (i < fullSize) embSqG(i) else outSqG(i - fullSize) }
  }
  
  def copy() : EmbedAdaGradUpdater = {
    val cc = new EmbedAdaGradUpdater(initialLearningRate,embSqG, outSqG)
    // important state
    cc.compressedEmbSqG_=(this.compressedEmbSqG)
    cc.compressedOutSqG_=(this.compressedEmbSqG)
    cc.isCompressed_=(this.isCompressed)
    cc
  }
  
  def compose(u: EmbedAdaGradUpdater) : EmbedAdaGradUpdater = {
    if (isCompressed) {
      val (eAr, eVls) = compressedEmbSqG
      val (oAr, oVls) = compressedOutSqG
      val (u_eAr, u_eVls) = u.compressedEmbSqG
      val (u_oAr, u_oVls) = u.compressedOutSqG
      var i = 0; while (i < eAr.length) {
        eAr(i) = math.max(eAr(i), u_eAr(i)).toByte
        oAr(i) = math.max(oAr(i), u_oAr(i)).toByte
        i += 1
      }
      i = 0; while (i < oVls.length) {
        eVls(i) = math.max(eVls(i), u_eVls(i))
        oVls(i) = math.max(oVls(i), u_oVls(i))
        i += 1
      }
      this
    } else {
      var i = 0; while (i < fullSize) {
        embSqG(i) = math.max(embSqG(i),u.embSqG(i))
        outSqG(i) = math.max(outSqG(i),u.outSqG(i))
        i += 1
      }
      this
    }
  }
  
  @inline
  final private def fastSqrt(x: Float) : Float = 
    java.lang.Float.intBitsToFloat(532483686 + (java.lang.Float.floatToRawIntBits(x) >> 1))
    
  final def updateEmbeddingSqG(i: Int, wArr: Array[Float], g: Float) = {    
    embSqG(i) += g * g
    wArr(i) += (initialLearningRate * g / (initialLearningRate + fastSqrt(embSqG(i))))
  }
  
  final def updateOutputSqG(i: Int, wArr: Array[Float], g: Float) = {
    outSqG(i) += g * g
    wArr(i) += (initialLearningRate * g / (initialLearningRate + fastSqrt(outSqG(i))))
  }  
  def updateNumProcessed() = {}
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
*/
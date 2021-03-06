package org.mitre.mandolin.util

import xerial.larray._

/**
 * A simple class to compress an array of floats in a very simple/crude manner.
 * A range of floats is divided into 256 'bins' each a fixed value apart. 
 */
object SimpleFloatCompressor {

  def binSearchGetBin(values: Array[Float], v: Float) : Int = {
    val vl = values.length
    var ind = vl / 2
    var found = false
    var lastLowInd = 0
    var lastHighInd = vl
    var tmp = 0
    while(!found) {
      if ((ind >= (vl-1)) || (ind == 0) || ((v >= values(ind)) && (v <= values(ind+1)))) found = true
      else {
        tmp = ind
        if (v > values(ind)) {
          lastLowInd = ind
          ind = ind + ((lastHighInd - ind) / 2)          
        }
        else if (v < values(ind)) {
          lastHighInd = ind
          ind = ind - (ind - lastLowInd) / 2 - 1
        }
        else found = true        
      }
    }
    ind
  }
  
  def compress(ar2: Simple2DArray) : (Simple2DByteArray, Array[Float]) = {
    var mxVal = -Double.MaxValue
    var mnVal = Double.MaxValue
    val rowLen = ar2.getDim1
    val colLen = ar2.getDim2
    val values = Array.fill(256)(0.0f)
    var i = 0; while (i < rowLen) {
      var j = 0; while (j < colLen) {
        val v = ar2(i,j)
        if (v > mxVal) mxVal = v
        if (v < mnVal) mnVal = v
        j += 1
      }
      i += 1
    }
    val binSize = (mxVal - mnVal).toFloat / 256
    i = 0; while(i < 256) {
      values(i) = mnVal.toFloat + binSize * i
      i += 1
    }
    val barray = Simple2DArray.byteArray(rowLen, colLen)
    (barray,values)
  }
  
  def compress(ar: LArray[Float]) : (LArray[Byte], Array[Float]) = {
    var mxVal = -Double.MaxValue
    var mnVal = Double.MaxValue
    val ll = ar.length
    val values = Array.fill(256)(0.0f)
    var i = 0; while (i < ll) {
      val v = ar(i)
      if (v > mxVal) mxVal = v
      if (v < mnVal) mnVal = v
      i += 1
    }
    val binSize = (mxVal - mnVal).toFloat / 256
    i = 0; while(i < 256) {
      values(i) = mnVal.toFloat + binSize * i
      i += 1
    }
    val bArray = LArray.of[Byte](ll)
    i = 0; while (i < ll) {
      val ind = binSearchGetBin(values, ar(i))
      bArray(i) = (ind-128).toByte
      i += 1
    }
    (bArray, values)
  }
  
  def compress(ar: Array[Float]) : (Array[Byte], Array[Float]) = {
    var mxVal = -Double.MaxValue
    var mnVal = Double.MaxValue
    val ll = ar.length
    val values = Array.fill(256)(0.0f)
    var i = 0; while (i < ll) {
      val v = ar(i)
      if (v > mxVal) mxVal = v
      if (v < mnVal) mnVal = v
      i += 1
    }
    val binSize = (mxVal - mnVal).toFloat / 256
    i = 0; while(i < 256) {
      values(i) = mnVal.toFloat + binSize * i
      i += 1
    }
    val bArray = Array.fill[Byte](ll)(0)
    i = 0; while (i < ll) {
      val ind = binSearchGetBin(values, ar(i))
      bArray(i) = (ind-128).toByte
      i += 1
    }
    (bArray, values)
  }
}
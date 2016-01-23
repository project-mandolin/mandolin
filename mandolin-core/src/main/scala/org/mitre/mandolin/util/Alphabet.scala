package org.mitre.mandolin.util
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

/**
 * Facilitates symbol tables and mappings from surface forms to indexed items
 */
abstract class Alphabet {

  def ofString(s: String): Int
  def ofString(s: String, v: Double) : Int
  
  /*
   * This provides a hook into an Alphabet so that it may keep scaling factors or other
   * information that can be used to modify a raw feature value.
   */
  def getValue(f: Int, v: Double) : Double
  def getSize: Int
  def ensureFixed: Unit  
  def ensureUnFixed : Unit
  def getInverseMapping : Map[Int,String]
  def getMapping : collection.mutable.HashMap[String, Int]
}

/**
 * Alphabet that assumes that each feature name is actually the index
 * that corresponds to that feature. This is ONE based by default.
 * @param size Current size of the alphabet
 * @author wellner
 */
class IdentityAlphabet(var size: Int, val oneBased: Boolean = true) extends Alphabet with Serializable {

  var fixed = false
  def this() = this(0)

  def getValue(f: Int, v: Double) = v
  def ofString(s: String, vl: Double) : Int = ofString(s)
  def ofString(s: String): Int = try {
    val i = if (oneBased) s.toInt - 1 else s.toInt
    if (!fixed && (i >= size)) {
      size = i + 1
    } 
    i
  } catch { case _: Throwable => throw new RuntimeException("Feature index expected as integer, found: " + s) }

  def getSize = size
  def ensureFixed = { fixed = true}
  def ensureUnFixed = { fixed = false}
  def getInverseMapping = throw new RuntimeException("Inverse not available from Identity Alphabet")
  def getMapping = throw new RuntimeException("Mapping not available from Identity Alphabet")
}

class IdentityAlphabetWithUnitScaling(zeroMaxMin: Boolean, s: Int) extends IdentityAlphabet(s) with Serializable {
  def this(s: Int) = this(true, s)
  val minVals = new collection.mutable.HashMap[Int, Double]
  val maxVals = new collection.mutable.HashMap[Int, Double]
  
  var totalMax = 0.0
  var totalMin = 0.0
  
  lazy val fmin = Array.tabulate(this.getSize){i => minVals.get(i).getOrElse(0.0)}
  lazy val fmax = Array.tabulate(this.getSize){i => maxVals.get(i).getOrElse(0.0)}
      
  override def getValue(fid: Int, v: Double) = {
    val mm = fmin(fid-1) // because of being one-based
    val mx = fmax(fid-1)
    val fv = v    
    if (mm == mx) { // this happens if we didn't see the feature when building the alphabet
      // normalize these features using the total max and min values across all features in training set
      if (fv > totalMax) 1.0 else if (fv > totalMin) (fv - totalMin) / (totalMax - totalMin) else 0.0
    } else {
      if (fv > mx) 1.0 else if (fv > mm) (fv - mm) / (mx - mm) else 0.0
    }
  }
  
  override def ofString(s: String, ivl: Double) : Int = {
    val i = super.ofString(s,ivl)
    val vlMin = if (zeroMaxMin) math.min(0.0,ivl) else ivl 
    val vlMax = ivl
    if (ivl > totalMax) totalMax = ivl
    if (ivl < totalMin) totalMin = ivl
    if (minVals.contains(i)) {
      if (vlMin < minVals(i)) minVals.update(i,vlMin)      
    } else minVals.update(i,vlMin)
    if (maxVals.contains(i)) {
      if (vlMax > maxVals(i)) maxVals.update(i,vlMax)
    } else maxVals.update(i,vlMax)
    i
  }
  
}

/**
 * Standard alphabet implemented using a `scala.collection.mutable.HashMap[String,Int]` 
 */
class StdAlphabet extends Alphabet with Serializable {
  val mapping = new collection.mutable.HashMap[String, Int]  
  var curSize = 0
  var fixed = false
  def ofString(s: String) = ofString(s, 1.0)
  def getValue(f: Int, v: Double) = v
  def ofString(s: String, vl: Double): Int = {
    mapping.get(s) match {
      case Some(v) => v
      case None =>
        if (fixed) -1 else {
          val i = curSize
          mapping.update(s, i)
          curSize += 1
          i
        }
    }
  }
  def ensureFixed = { fixed = true }
  def ensureUnFixed = { fixed = false }
  def getSize = curSize
  def getInverseMapping = {
    var inv = Map[Int,String]()
    mapping.foreach{ case (s,i) => inv += ((i,s))}
    inv
  }
  def getMapping = mapping
}

/**
 * A standard alphabet that facilitates unit scaling of the input features.
 * As the alphabet is constructed via an initial pass over the input data; the maximum
 * and minimum values are recorded so that feature values can be unit scaled
 * on a second pass of instantiating the feature vectors.
 * @author wellner
 */
class AlphabetWithUnitScaling(zeroMaxMin: Boolean = true) extends StdAlphabet {
  val minVals = new collection.mutable.HashMap[Int, Double]
  val maxVals = new collection.mutable.HashMap[Int, Double]
  
  lazy val fmin = Array.tabulate(this.getSize){i => minVals(i)}
  lazy val fmax = Array.tabulate(this.getSize){i => maxVals(i)}
      
  override def getValue(fid: Int, v: Double) = {
    val mm = fmin(fid)
    val mx = fmax(fid)
    val fv = v    
    if (fv > mx) 1.0 else if (fv > mm) (fv - mm) / (mx - mm) else 0.0
  }
  
  override def ofString(s: String, ivl: Double) : Int = {
    val i = super.ofString(s,ivl)
    val vlMin = if (zeroMaxMin) math.min(0.0,ivl) else ivl 
    val vlMax = ivl
    if (minVals.contains(i)) {
      if (vlMin < minVals(i)) minVals.update(i,vlMin)      
    } else minVals.update(i,vlMin)
    if (maxVals.contains(i)) {
      if (vlMax > maxVals(i)) maxVals.update(i,vlMax)
    } else maxVals.update(i,vlMax)
    i
  }
}

class PrescaledAlphabet(minVals: Array[Double], maxVals: Array[Double]) extends StdAlphabet {
  override def getValue(fid: Int, v: Double) = {
    val mm = minVals(fid)
    val mx = maxVals(fid)
    val fv = v    
    if (fv > mx) 1.0 else if (fv > mm) (fv - mm) / (mx - mm) else 0.0
  }
}

/**
 * A Random or Hashing alphabet that uses a Murmur hash algorithm to map
 * input strings to integers efficiently but with chances for collision.
 * @param size the size of the hash/mod space 
 */
class RandomAlphabet(size: Int) extends Alphabet with Serializable {

  def ensureFixed = {}
  def ensureUnFixed = {}
  def getValue(f: Int, v: Double) = v
  @inline
  final def ofString(s: String): Int = { update(scala.util.hashing.MurmurHash3.stringHash(s), size) }
  
  @inline
  final def ofString(s: String, v: Double): Int = ofString(s)
  
  def getSize = size
  def getInverseMapping = throw new RuntimeException("Inverse not available from Random Alphabet")
  def getMapping = throw new RuntimeException("Mapping not available from Random Alphabet")
  
  @inline
  final def update(k: Int, size: Int): Int = {
    if (k < 0) (math.abs(k) % size).toInt
    else (k % size).toInt
  }
}

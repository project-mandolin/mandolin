package org.mitre.mandolin.util
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.gm.{Feature, NonUnitFeature}

/**
 * Trait that provides useful utilities for parsing feature vectors in sparse
 * or dense representations
 * @author wellner
 */
trait LineParser {
  
  def checkFeature(s: String, prefFilter: Option[String] = None) = {
      if (prefFilter.isDefined) {
        s.startsWith(prefFilter.get) // check whether feature begins with a sub-string
      } else true
    }

    
  def getSparseInputVec(li: List[String], mp: Alphabet, buildVecs: Boolean = true,
    prefFilter: Option[String] = None) = {
    val fvbuf = new collection.mutable.ArrayBuffer[Feature]
        li foreach { f =>
          f.split(':').toList match {
            case a :: b :: Nil =>
              if (checkFeature(a)) {
                if (buildVecs) {
                  val bd = b.toDouble
                  val id = mp.ofString(a, bd)
                  if (id >= 0) fvbuf append (new NonUnitFeature(mp.ofString(a, bd), bd))
                } else mp.ofString(a, b.toDouble)
              }
            case a :: Nil =>
              if (checkFeature(a)) {
                if (buildVecs) {
                  val id = mp.ofString(a)
                  if (id >= 0) fvbuf append (new Feature(id))
                } else mp.ofString(a)
              }
            case a => throw new RuntimeException("Unparsable feature: " + a)
          }
        }
    fvbuf.toArray
  }
  
  def sparseOfLine(s: String, mp: Alphabet, sep: Char = ' ', buildVecs: Boolean = true,
    prefFilter: Option[String] = None, noLabels: Boolean = false): (Option[String], Array[Feature], Option[String]) = {

    val (vec, id) = s.split('#').toList match { 
      case a :: b :: Nil => (a, Some(b)) 
      case a :: _ => (a, None) 
      case Nil => throw new RuntimeException("Ill-formed input vector: " + s)}
    
    if (!noLabels) {
      vec.split(sep).toList match {
        case lab :: rest =>
          val fvInput = getSparseInputVec(rest, mp, buildVecs, prefFilter)
          (Some(lab), fvInput, id)
        case l => throw new RuntimeException("Unparsable Line: " + l)
      }
    } else {
      val fvInput = getSparseInputVec(vec.split(sep).toList, mp, buildVecs, prefFilter)
      (None, fvInput, id)
    }
        
  }

  /** Parse dense line.
   *  @return a 2-tuple: a ''label'' and an array of `Double` primitive type elements */
  def denseOfLine(s: String, dim: Int, sep: Char = ' ', noLabels: Boolean = false): (Option[String], Array[Double]) = {
    val elements = s.split(sep)
    if (!noLabels) {
      assert(elements.length == dim + 1)
      val lab = elements(0)
      (Some(lab), Array.tabulate(dim) { i => elements(i + 1).toDouble })  
    } else {
      assert(elements.length == dim)
      (None, Array.tabulate(dim) { i => elements(i + 1).toDouble })
    }
    
  }
}

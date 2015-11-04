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
  def sparseOfLine(s: String, mp: Alphabet, sep: Char = ' ', addBias: Boolean = true, buildVecs: Boolean = true,
    prefFilter: Option[String] = None): (String, Array[Feature], Option[String]) = {

    def checkFeature(s: String) = {
      if (prefFilter.isDefined) {
        s.startsWith(prefFilter.get) // check whether feature begins with a sub-string
      } else true
    }

    val (vec, id) = s.split('#').toList match { 
      case a :: b :: Nil => (a, Some(b)) 
      case a :: _ => (a, None) 
      case Nil => throw new RuntimeException("Ill-formed input vector: " + s)}
    
    vec.split(sep).toList match {
      case lab :: rest =>
        val fvbuf = new collection.mutable.ArrayBuffer[Feature]
        if (addBias) {
          if (buildVecs) fvbuf append (new NonUnitFeature(mp.ofString("=BIAS="), 1.0)) // not very efficient
          else mp.ofString("=BIAS=")
        }
        rest foreach { f =>
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
        (lab, fvbuf.toArray, id)
      case l => throw new RuntimeException("Unparsable Line: " + l)
    }
  }

  /** Parse dense line.
   *  @return a 2-tuple: a ''label'' and an array of `Double` primitive type elements */
  def denseOfLine(s: String, dim: Int, sep: Char = ' '): (String, Array[Double]) = {
    val elements = s.split(sep)
    assert(elements.length == dim + 1)
    val lab = elements(0)
    (lab, Array.tabulate(dim) { i => elements(i + 1).toDouble })
  }
}

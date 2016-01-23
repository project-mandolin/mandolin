package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */
import org.mitre.mandolin.util.{DenseTensor1 => DenseVec, Alphabet}

/**
 * Reads simple, dense feature vector representations from file
 * @author wellner
 */
class DenseReader(val delim: Char) extends Serializable {
  
  def getLabeledLine(l: String) : (String, DenseVec) = {
    val items = l.split(delim)
    val vec = DenseVec.tabulate(items.length - 1){i => items(i+1).toFloat }
    (items(0), vec)
  }
  
  def getUnlabeledLine(l: String) : DenseVec = {
    val items = l.split(delim)
    DenseVec.tabulate(items.length - 1){i => items(i+1).toFloat }
  }
}

/**
 * Reader that maps a sparse input representation to a `DenseVec` 
 * where the feature indices for each feature have already been computed.
 * E.g. `<label> 1:1.0 9:1.0 10:0.95 ... `
 * @author wellner
 */
class SparseToDenseReader(val delim: Char, fa: Alphabet, val dim: Int) extends Serializable {
  def getLabeledLine(l: String) : (String, DenseVec) = {
    val items = l.split(delim)
    val vec : DenseVec = DenseVec.zeros(dim)
    var i = 0;
    val label = items(0)
    while (i < items.length - 1) {
      val el = items(i+1).split(':')
      // the -1 below assumes input has 1-based indexing
      val vv = fa.getValue(el(0).toInt, el(1).toFloat)
      vec(el(0).toInt - 1) = vv.toFloat
      i += 1
    }
    (label, vec)
  }
}

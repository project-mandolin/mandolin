package org.mitre.mandolin.mlp

/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.util.{Alphabet, IdentityAlphabet}
import org.mitre.mandolin.util.{LineParser, DenseTensor1 => DenseVec, SparseTensor1 => SparseVec, Tensor1 => Vec}

/**
  * Maps input lines in a sparse-vector format `<label> <f1>:<val> <f2>:<val> ... `
  * to a dense vector input. This is most efficient if the inputs aren't ''too'' sparse
  * and the dimensionality isn't too great.
  *
  * @param alphabet Feature alphabet
  * @param la       Label alphabet
  * @author wellner
  */
class VecFeatureExtractor(alphabet: Alphabet, la: Alphabet)
  extends FeatureExtractor[String, MMLPFactor] with LineParser with Serializable {
  var ind = 0

  def getAlphabet = alphabet

  def extractFeatures(s: String): MMLPFactor = {
    val (l, spv, id) = sparseOfLine(s, alphabet, noLabels = noLabels)
    val dVec: DenseVec = DenseVec.zeros(alphabet.getSize)
    spv foreach { f =>
      if (f.fid >= 0) {
        val fv = alphabet.getValue(f.fid, f.value).toFloat
        dVec.update(f.fid, fv)
      }
    }
    val l_ind = la.ofString(l)
    val lv = DenseVec.zeros(la.getSize)
    lv.update(l_ind, 1.0f) // one-hot encoding
    ind += 1
    new StdMMLPFactor(ind, dVec, lv, id)
  }

  def getNumberOfFeatures = alphabet.getSize
}

/**
  * Maps input lines in a sparse-vector format `<label> <f1>:<val> <f2>:<val> ... `
  * to a ''sparse'' vector input. This is most efficient if the inputs have high dimensionality
  * and are sparse.
  *
  * @param alphabet Feature alphabet
  * @param la       Label alphabet
  * @author wellner
  */
class SparseVecFeatureExtractor(alphabet: Alphabet, la: Alphabet)
  extends FeatureExtractor[String, MMLPFactor] with LineParser with Serializable {
  var ind = 0

  
  def getAlphabet = alphabet

  def extractFeatures(s: String): MMLPFactor = {
    val (l, spv, id) = sparseOfLine(s, alphabet, noLabels=noLabels)
    val spVec: SparseVec = SparseVec(alphabet.getSize)
    spv foreach { f =>
      if (f.fid >= 0) {
        val fv = alphabet.getValue(f.fid, f.value).toFloat
        spVec.update(f.fid, fv)
      }
    }
    val l_ind = la.ofString(l)
    val lv = SparseVec.getOneHot(la.getSize, l_ind)
    ind += 1
    new SparseMMLPFactor(ind, spVec.asStatic, lv, id)
  }

  def getNumberOfFeatures = alphabet.getSize
}

/**
  * Extractor that constructs `DenseVec` dense vectors from an
  * input sparse representation where the feature indices for each feature have already been computed.
  * E.g. `<label> 1:1.0 9:1.0 10:0.95 ... `
  * This is most efficient and avoids using another symbol table if
  * the features have already been mapped to integers - e.g. with datasets in libSVM/libLINEAR format.
  *
  * @author
  */
class StdVectorExtractorWithAlphabet(la: Alphabet, fa: Alphabet, nfs: Int) 
extends FeatureExtractor[String, MMLPFactor] with Serializable {
  val reader = new SparseToDenseReader(' ', fa, nfs)
  
  def getAlphabet = fa

  def extractFeatures(s: String): MMLPFactor = {
    if (noLabels) {
      val features = reader.getUnlabeledLine(s)
      val targetVec = DenseVec.zeros(la.getSize)
      new StdMMLPFactor(features, targetVec)
    } else {
    val (lab, features) = reader.getLabeledLine(s)
    val targetVec = DenseVec.zeros(la.getSize)
    targetVec.update(la.ofString(lab), 1.0f) // set one-hot    
    new StdMMLPFactor(features, targetVec)
    }
  }

  def getNumberOfFeatures = nfs
}

class StdVectorExtractorRegression(fa: Alphabet, nfs: Int) 
extends FeatureExtractor[String, MMLPFactor] with Serializable {
  val reader = new SparseToDenseReader(' ', fa, nfs)


  def getAlphabet = fa

  def extractFeatures(s: String): MMLPFactor = {
    if (noLabels) {
      val features = reader.getUnlabeledLine(s)
      val targetVec = DenseVec.zeros(1)
      new StdMMLPFactor(features, targetVec)
    } else {
    val (lab, features) = reader.getLabeledLine(s)
    val lv = lab.toFloat
    val targetVec = DenseVec.tabulate(1) { _ => lv }
    new StdMMLPFactor(features, targetVec)
    }
  }

  def getNumberOfFeatures = nfs
}

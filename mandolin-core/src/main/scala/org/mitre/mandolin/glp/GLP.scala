package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.config.{MandolinMLPSettings, GeneralLearnerSettings}

import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.util.{Alphabet, AlphabetWithUnitScaling, StdAlphabet, IdentityAlphabet, IOAssistant}
import org.mitre.mandolin.gm.Feature
import org.mitre.mandolin.util.{LineParser, DenseTensor1 => DenseVec, SparseTensor1 => SparseVec, Tensor1 => Vec}



/**
 * Maps input lines in a sparse-vector format `<label> <f1>:<val> <f2>:<val> ... `
 * to a dense vector input. This is most efficient if the inputs aren't ''too'' sparse
 * and the dimensionality isn't too great.
 * @param alphabet Feature alphabet
 * @param la Label alphabet
 * @author wellner
 */
class VecFeatureExtractor(alphabet: Alphabet, la: Alphabet)
  extends FeatureExtractor[String, GLPFactor] with LineParser with Serializable {
  var ind = 0
  
  def getAlphabet = alphabet
  
  def extractFeatures(s: String): GLPFactor = {
    val (l, spv, id) = sparseOfLine(s, alphabet)
    val dVec : DenseVec = DenseVec.zeros(alphabet.getSize)    
    spv foreach { f =>
      if (f.fid >= 0) {
        val fv = alphabet.getValue(f.fid, f.value).toFloat
        dVec.update(f.fid, fv)
      }
    }
    val l_ind = la.ofString(l)
    val lv = DenseVec.zeros(la.getSize)
    lv.update(l_ind,1.0f) // one-hot encoding
    ind += 1
    new StdGLPFactor(ind, dVec, lv, id)    
  }
  def getNumberOfFeatures = alphabet.getSize
}

/**
 * Maps input lines in a sparse-vector format `<label> <f1>:<val> <f2>:<val> ... `
 * to a ''sparse'' vector input. This is most efficient if the inputs have high dimensionality
 * and are sparse. 
 * @param alphabet Feature alphabet
 * @param la Label alphabet
 * @author wellner
 */
class SparseVecFeatureExtractor(alphabet: Alphabet, la: Alphabet)
  extends FeatureExtractor[String, GLPFactor] with LineParser with Serializable {
  var ind = 0
  
  def getAlphabet = alphabet
  
  def extractFeatures(s: String): GLPFactor = {
    val (l, spv, id) = sparseOfLine(s, alphabet)
    val spVec : SparseVec = SparseVec(alphabet.getSize)    
    spv foreach { f =>
      if (f.fid >= 0) {
        val fv = alphabet.getValue(f.fid, f.value).toFloat
        spVec.update(f.fid, fv)
      }
    }
    val l_ind = la.ofString(l)
    val lv = SparseVec.getOneHot(la.getSize, l_ind)
    ind += 1
    new SparseGLPFactor(ind, spVec.asStatic, lv, id)    
  }
  def getNumberOfFeatures = alphabet.getSize
}

/**
 * Extractor that constructs `DenseVec` dense vectors from an
 * input sparse representation where the feature indices for each feature have already been computed.
 * E.g. `<label> 1:1.0 9:1.0 10:0.95 ... `
 * This is most efficient and avoids using another symbol table if
 * the features have already been mapped to integers - e.g. with datasets in libSVM/libLINEAR format. 
 * @author
 */
class StdVectorExtractorWithAlphabet(la: Alphabet, fa: Alphabet, nfs: Int) extends FeatureExtractor[String, GLPFactor] with Serializable {
  val reader = new SparseToDenseReader(' ', fa, nfs)
  
  def getAlphabet = fa
  
  def extractFeatures(s: String) : GLPFactor = {
    val (lab, features) = reader.getLabeledLine(s)
    val targetVec = DenseVec.zeros(la.getSize)
    targetVec.update(la.ofString(lab), 1.0f) // set one-hot
    new StdGLPFactor(features, targetVec)  
  }
  def getNumberOfFeatures = nfs
}

class StdVectorExtractorRegression(fa: Alphabet, nfs: Int) extends FeatureExtractor[String, GLPFactor] with Serializable {
  val reader = new SparseToDenseReader(' ', fa, nfs)
  
  def getAlphabet = fa
  
  def extractFeatures(s: String) : GLPFactor = {
    val (lab, features) = reader.getLabeledLine(s)
    val lv = lab.toFloat
    val targetVec = DenseVec.tabulate(1){_ => lv}
    new StdGLPFactor(features, targetVec)  
  }
  def getNumberOfFeatures = nfs
}

class BagOneHotExtractor(la: Alphabet, nfs: Int) extends FeatureExtractor[String, GLPFactor] with Serializable {
  def getNumberOfFeatures = nfs
  def getAlphabet = new IdentityAlphabet(nfs)
  var x = 0
  def extractFeatures(s: String) : GLPFactor = {
    x += 1
    val line = s.split(" ")
    val ln = line.length
    val lab = line(0)
    var mm = Map[Int,Float]()
    var i = 1; while (i < ln) {
      val av = line(i).split(":")
      val ind = av(0).toInt
      val vl = if (av.length > 1) av(1).toDouble else 1.0
      val cv = mm.get(ind).getOrElse(0.0f)
      mm += (ind -> (cv + 1.0f))
      i += 1
    }
    val targetVec : Vec = SparseVec.getOneHot(la.getSize, la.ofString(lab))
    val (inds, vls) = mm.toList.unzip
    val spv = SparseVec.getStaticSparseTensor1(nfs, inds.toArray, vls.toArray)
    new SparseGLPFactor(spv, targetVec)
  }
}

class SequenceOneHotExtractor(la: Alphabet, nfs: Int) extends FeatureExtractor[String, GLPFactor] with Serializable {

  var x = 0
  def getNumberOfFeatures = nfs
  def getAlphabet = new IdentityAlphabet(nfs)
  def extractFeatures(s: String) : GLPFactor = {
    x += 1
    val line = s.split(" ")
    val ln = line.length
    val lab = line(0)
    var flist : List[(Int,Float)] = Nil
    var i = 1; while (i < ln) {
      val av = line(i).split(":")
      val ind = av(0).toInt + (i - 1) * nfs
      val vl = if (av.length > 1) av(1).toFloat else 1.0f
      flist = (ind, vl) :: flist      
      i += 1
    }
    val targetVec : Vec = SparseVec.getOneHot(la.getSize, la.ofString(lab))
    val (inds, vls) = flist.unzip
    if ((x % 10000) == 1) println("Processed " + x + " input vectors")
    new SparseGLPFactor(SparseVec.getStaticSparseTensor1(nfs * (ln - 1), inds.toArray, vls.toArray), targetVec)
  }
}

package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.optimize.{ModelWriter, Updater}
import com.esotericsoftware.kryo.io.{ Input => KInput, Output => KOutput }
import com.twitter.chill.{ EmptyScalaKryoInstantiator, AllScalaRegistrar }
import org.mitre.mandolin.util.{ Alphabet, IOAssistant, StdAlphabet }
import org.mitre.mandolin.util.{ LocalIOAssistant, IdentityAlphabet }
import org.mitre.mandolin.glp.local.LocalGLPOptimizer
import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.predict.local.LocalTrainer
import org.mitre.mandolin.optimize.local.LocalOptimizerEstimator

case class GLPModelSpec(wts: GLPWeights, ann: ANNetwork, 
    la: Alphabet, fe: FeatureExtractor[String, GLPFactor]) extends Serializable

abstract class GLPModelWriter {
  def writeModel(weights: GLPWeights) : Unit
  def writeModel(io: IOAssistant, filePath: String, w: GLPWeights, la: Alphabet, ann: ANNetwork, fe: FeatureExtractor[String, GLPFactor]) : Unit
}

abstract class GLPModelReader {
  def readModel(f: String, io: IOAssistant) : GLPModelSpec
}

object GLPTrainerBuilder extends AbstractProcessor {
  
  def apply(appSettings: MandolinMLPSettings) : (LocalTrainer[String, GLPFactor, GLPWeights], ANNetwork) = {
    val io = new LocalIOAssistant
    val components = getComponentsViaSettings(appSettings, io)
    val fe = components.featureExtractor
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(appSettings, components.ann)
    (new LocalTrainer(fe, optimizer), components.ann)
  } 
  
  def apply(mspec: IndexedSeq[LType], lines: Iterator[String]) : (LocalTrainer[String, GLPFactor, GLPWeights], ANNetwork) = {   
    val io = new LocalIOAssistant
    val la = new StdAlphabet
    val gset = getComponentsInducedAlphabet(mspec, lines: Iterator[String], la: Alphabet, false, -1, io)
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(new MandolinMLPSettings ,gset.ann)
    (new LocalTrainer(gset.featureExtractor, optimizer), gset.ann)
  }
  
  /**
   * This method creates a local trainer without a feature extractor. Will be used on feature vectors
   * already constructed as GLPFactor objects.
   */
  def apply(modelSpec: IndexedSeq[LType]) : (LocalTrainer[String, GLPFactor, GLPWeights], ANNetwork) = {
    val (nn, predictor, oc) = getSubComponents(modelSpec)
    val settings = new MandolinMLPSettings
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(settings, nn)
    (new LocalTrainer(optimizer), nn)
  }
  
  def apply[T](modelSpec: IndexedSeq[LType], fe: FeatureExtractor[T,GLPFactor], idim: Int, odim: Int) : (LocalTrainer[T, GLPFactor, GLPWeights], ANNetwork) = {
    val (nn, predictor, oc) = getSubComponents(modelSpec, idim, odim)
    val settings = new MandolinMLPSettings
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(settings, nn)
    (new LocalTrainer(fe, optimizer), nn)
  }
  
  def apply[T](modelSpec: IndexedSeq[LType], fe: FeatureExtractor[T,GLPFactor], idim: Int, odim: Int, sets: Seq[(String, Any)]) : (LocalTrainer[T, GLPFactor, GLPWeights], ANNetwork) = {
    val (nn, predictor, oc) = getSubComponents(modelSpec, idim, odim)
    val settings = (new MandolinMLPSettings).withSets(sets)
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(settings, nn)
    (new LocalTrainer(fe, optimizer), nn)
  }
  
  /**
   * This method allows for an arbitrary feature extractor to be used with an arbitrary model spec
   */
  def apply[T](modelSpec: IndexedSeq[LType], fe: FeatureExtractor[T, GLPFactor]) : (LocalTrainer[T, GLPFactor, GLPWeights], ANNetwork) = {
    val (nn, predictor, oc) = getSubComponents(modelSpec)
    val settings = new MandolinMLPSettings
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(settings, nn)
    (new LocalTrainer(fe, optimizer), nn)
  }
  

  def apply(modelSpec: IndexedSeq[LType], fa: Alphabet, la: Alphabet, sparse: Boolean = false) : (LocalTrainer[String, GLPFactor, GLPWeights], ANNetwork) = {
    val (nn, predictor, oc) = getSubComponents(modelSpec)    
    val fe = if (sparse)
      new SparseVecFeatureExtractor(la, fa)
    else {
      val numInputs = modelSpec.head.dim
      new StdVectorExtractorWithAlphabet(la, fa, numInputs)
    }
    val settings = new MandolinMLPSettings
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(settings, nn)
    (new LocalTrainer(fe, optimizer), nn)
  }
  
  // XXX - let API here simply create the appropriate settings; seems backwards but most convenient
  def apply(modelSpec: IndexedSeq[LType], sets: Seq[(String, Any)]) : (LocalTrainer[String, GLPFactor, GLPWeights], ANNetwork) = {
    val settings = (new MandolinMLPSettings).withSets(sets)
    val (nn, predictor, oc) = getSubComponents(modelSpec)
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(settings, nn)
    (new LocalTrainer(optimizer), nn)
  }

}


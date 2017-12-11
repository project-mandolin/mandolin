package org.mitre.mandolin.mlp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.util.{ Alphabet, IOAssistant, StdAlphabet }
import org.mitre.mandolin.util.LocalIOAssistant
import org.mitre.mandolin.mlp.standalone.MMLPOptimizer
import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.predict.standalone.Trainer

case class MMLPModelSpec(wts: MMLPWeights, ann: ANNetwork,
                         la: Alphabet, fe: FeatureExtractor[String, MMLPFactor]) extends Serializable

abstract class MMLPModelWriter {
  def writeModel(weights: MMLPWeights) : Unit
  def writeModel(io: IOAssistant, filePath: String, w: MMLPWeights, la: Alphabet, ann: ANNetwork, fe: FeatureExtractor[String, MMLPFactor]) : Unit
}

abstract class MMLPModelReader {
  def readModel(f: String, io: IOAssistant) : MMLPModelSpec
}

object MMLPTrainerBuilder extends AbstractProcessor {
  
  def apply(appSettings: MandolinMLPSettings) : (Trainer[String, MMLPFactor, MMLPWeights], ANNetwork) = {
    val io = new LocalIOAssistant
    val components = getComponentsViaSettings(appSettings, io)
    val fe = components.featureExtractor
    val optimizer = MMLPOptimizer.getOptimizer(appSettings, components.ann)
    (new Trainer(fe, optimizer), components.ann)
  } 
  
  def apply(mspec: IndexedSeq[LType], lines: Iterator[String]) : (Trainer[String, MMLPFactor, MMLPWeights], ANNetwork) = {
    val io = new LocalIOAssistant
    val la = new StdAlphabet
    val gset = getComponentsInducedAlphabet(mspec, lines: Iterator[String], la: Alphabet, false, -1, io)
    val optimizer = MMLPOptimizer.getOptimizer(new MandolinMLPSettings ,gset.ann)
    (new Trainer(gset.featureExtractor, optimizer), gset.ann)
  }
  
  /**
   * This method creates a standalone trainer without a feature extractor. Will be used on feature vectors
   * already constructed as MMLPFactor objects.
   */
  def apply(modelSpec: IndexedSeq[LType]) : (Trainer[String, MMLPFactor, MMLPWeights], ANNetwork) = {
    val (nn, predictor, oc) = getSubComponents(modelSpec, false)
    val settings = new MandolinMLPSettings
    val optimizer = MMLPOptimizer.getOptimizer(settings, nn)
    (new Trainer(optimizer), nn)
  }
  
  def apply[T](modelSpec: IndexedSeq[LType], fe: FeatureExtractor[T,MMLPFactor], idim: Int, odim: Int) : (Trainer[T, MMLPFactor, MMLPWeights], ANNetwork) = {
    val (nn, predictor, oc) = getSubComponents(modelSpec, idim, odim, (odim < 2))
    val settings = new MandolinMLPSettings
    val optimizer = MMLPOptimizer.getOptimizer(settings, nn)
    (new Trainer(fe, optimizer), nn)
  }
  
  def apply[T](modelSpec: IndexedSeq[LType], fe: FeatureExtractor[T,MMLPFactor], idim: Int, odim: Int, sets: Seq[(String, Any)]) : (Trainer[T, MMLPFactor, MMLPWeights], ANNetwork) = {
    val (nn, predictor, oc) = getSubComponents(modelSpec, idim, odim, (odim < 2))
    val settings = (new MandolinMLPSettings).withSets(sets)
    val optimizer = MMLPOptimizer.getOptimizer(settings, nn)
    (new Trainer(fe, optimizer), nn)
  }
  
  /**
   * This method allows for an arbitrary feature extractor to be used with an arbitrary model spec
   */
  def apply[T](modelSpec: IndexedSeq[LType], fe: FeatureExtractor[T, MMLPFactor]) : (Trainer[T, MMLPFactor, MMLPWeights], ANNetwork) = {
    val (nn, predictor, oc) = getSubComponents(modelSpec, false)
    val settings = new MandolinMLPSettings
    val optimizer = MMLPOptimizer.getOptimizer(settings, nn)
    (new Trainer(fe, optimizer), nn)
  }
  
  /**
   * This method allows for an arbitrary feature extractor to be used with an arbitrary model spec
   */
  def apply[T](modelSpec: IndexedSeq[LType], fe: FeatureExtractor[T, MMLPFactor], reg: Boolean) : (Trainer[T, MMLPFactor, MMLPWeights], ANNetwork) = {
    val (nn, predictor, oc) = getSubComponents(modelSpec, reg)
    val settings = new MandolinMLPSettings
    val optimizer = MMLPOptimizer.getOptimizer(settings, nn)
    (new Trainer(fe, optimizer), nn)
  }
  

  def apply(modelSpec: IndexedSeq[LType], fa: Alphabet, la: Alphabet, sparse: Boolean = false, reg: Boolean = false) : (Trainer[String, MMLPFactor, MMLPWeights], ANNetwork) = {
    val (nn, predictor, oc) = getSubComponents(modelSpec, reg)    
    val fe = if (sparse)
      new SparseVecFeatureExtractor(la, fa)
    else {
      val numInputs = modelSpec.head.dim
      new StdVectorExtractorWithAlphabet(la, fa, numInputs)
    }
    val settings = new MandolinMLPSettings
    val optimizer = MMLPOptimizer.getOptimizer(settings, nn)
    (new Trainer(fe, optimizer), nn)
  }
  
  // XXX - let API here simply create the appropriate settings; seems backwards but most convenient
  def apply(modelSpec: IndexedSeq[LType], sets: Seq[(String, Any)]) : (Trainer[String, MMLPFactor, MMLPWeights], ANNetwork) = {
    val settings = (new MandolinMLPSettings).withSets(sets)
    val (nn, predictor, oc) = getSubComponents(modelSpec, settings.regression)
    val optimizer = MMLPOptimizer.getOptimizer(settings, nn)
    (new Trainer(optimizer), nn)
  }

}


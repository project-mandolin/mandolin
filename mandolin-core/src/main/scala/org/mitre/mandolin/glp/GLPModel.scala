package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.optimize.{ModelWriter, Updater}
import com.esotericsoftware.kryo.io.{ Input => KInput, Output => KOutput }
import com.twitter.chill.{ EmptyScalaKryoInstantiator, AllScalaRegistrar }
import org.mitre.mandolin.util.{ Alphabet, IOAssistant }
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
  
  def apply(appSettings: GLPModelSettings) : (LocalTrainer[String, GLPFactor, GLPWeights], ANNetwork) = {
    val io = new LocalIOAssistant
    val components = getComponentsViaSettings(appSettings, io)
    val fe = components.featureExtractor
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(appSettings, components.ann)
    (new LocalTrainer(fe, optimizer), components.ann)
  } 
  
  /**
   * This method creates a local trainer without a feature extractor. Will be used on feature vectors
   * already constructed as GLPFactor objects.
   */
  def apply(modelSpec: IndexedSeq[LType]) : (LocalTrainer[String, GLPFactor, GLPWeights], ANNetwork) = {
    val (nn, predictor, oc) = getSubComponents(modelSpec)
    val settings = new GLPModelSettings
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(settings, nn)
    (new LocalTrainer(optimizer), nn)
  }
  
  def apply(modelSpec: IndexedSeq[LType], fe: FeatureExtractor[String, GLPFactor]) : (LocalTrainer[String, GLPFactor, GLPWeights], ANNetwork) = {
    val (nn, predictor, oc) = getSubComponents(modelSpec)
    val numInputs = modelSpec.head.dim
    val fa = new IdentityAlphabet(numInputs)
    val labelAlphabet = new IdentityAlphabet(modelSpec.last.dim)
    val fe = new StdVectorExtractorWithAlphabet(labelAlphabet, fa, numInputs)
    val settings = new GLPModelSettings
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
    val settings = new GLPModelSettings
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(settings, nn)
    (new LocalTrainer(fe, optimizer), nn)
  }
  
  // XXX - let API here simply create the appropriate settings; seems backwards but most convenient
  def apply(modelSpec: IndexedSeq[LType], sets: Seq[(String, Any)]) : (LocalTrainer[String, GLPFactor, GLPWeights], ANNetwork) = {
    val settings = (new GLPModelSettings).withSets(sets)
    val (nn, predictor, oc) = getSubComponents(modelSpec)
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(settings, nn)
    (new LocalTrainer(optimizer), nn)
  }

}


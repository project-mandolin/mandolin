package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.optimize.{ModelWriter, Updater}
import com.esotericsoftware.kryo.io.{ Input => KInput, Output => KOutput }
import com.twitter.chill.{ EmptyScalaKryoInstantiator, AllScalaRegistrar }
import org.mitre.mandolin.util.{ Alphabet, IOAssistant }
import org.mitre.mandolin.util.{ LocalIOAssistant }
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

class GLPModelTrainer(_fe: FeatureExtractor[String, GLPFactor], _opt: LocalOptimizerEstimator[GLPFactor,GLPWeights]) 
extends LocalTrainer[String, GLPFactor, GLPWeights](_fe, _opt) {
  
  
}

object GLPTrainerBuilder extends AbstractProcessor {
  
  def apply(appSettings: GLPModelSettings) : LocalTrainer[String, GLPFactor, GLPWeights] = {
    val io = new LocalIOAssistant
    val components = getComponentsViaSettings(appSettings, io)
    val fe = components.featureExtractor
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(appSettings, components.ann)
    new LocalTrainer(fe, optimizer)
  } 
  
  /*
  def apply(modelSpec: IndexedSeq[LType]) : LocalTrainer[String, GLPFactor, GLPWeights] = {
    
  }
  * 
  */
}


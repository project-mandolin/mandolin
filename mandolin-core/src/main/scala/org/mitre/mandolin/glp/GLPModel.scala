package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.optimize.{ModelWriter, Updater}
import com.esotericsoftware.kryo.io.{ Input => KInput, Output => KOutput }
import com.twitter.chill.{ EmptyScalaKryoInstantiator, AllScalaRegistrar }
import org.mitre.mandolin.util.{ Alphabet, IOAssistant }
import org.mitre.mandolin.transform.FeatureExtractor

case class GLPModelSpec(wts: GLPWeights, ann: ANNetwork, 
    la: Alphabet, fe: FeatureExtractor[String, GLPFactor]) extends Serializable

abstract class GLPModelWriter {
  def writeModel(weights: GLPWeights) : Unit
  def writeModel(io: IOAssistant, filePath: String, w: GLPWeights, la: Alphabet, ann: ANNetwork, fe: FeatureExtractor[String, GLPFactor]) : Unit
}

abstract class GLPModelReader[U <: Updater[GLPWeights, GLPLossGradient, U]] {
  def readModel(f: String, io: IOAssistant) : GLPModelSpec
}


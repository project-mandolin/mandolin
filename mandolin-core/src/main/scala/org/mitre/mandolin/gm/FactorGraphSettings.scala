package org.mitre.mandolin.gm

import org.mitre.mandolin.glp.MandolinMLPSettings

class FactorGraphSettings(args: Array[String]) extends MandolinMLPSettings(args) {

  val singletonFile = asStr("mandolin.gm.singleton-file")
  val factorFile = asStr("mandolin.gm.factor-file")

  val singletonTestFile = asStrOpt("mandolin.gm.singleton-test-file")
  val factorTestFile = asStrOpt("mandolin.gm.factor-test-file")

  val subGradEpochs = asInt("mandolin.gm.subgrad-epochs")
  val sgAlpha = asDouble("mandolin.gm.sg-alpha")
  val inferAlgorithm = asStrOpt("mandolin.gm.infer-algorithm")
}
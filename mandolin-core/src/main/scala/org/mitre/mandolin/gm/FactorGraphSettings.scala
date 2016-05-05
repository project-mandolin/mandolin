package org.mitre.mandolin.gm

import org.mitre.mandolin.glp.GLPModelSettings

class FactorGraphSettings(args: Array[String]) extends GLPModelSettings(args) {
  
  val singletonFile = asStr("mandolin.gm.singleton-file")
  val factorFile = asStr("mandolin.gm.factor-file")
  
  val singletonTestFile = asStr("mandolin.gm.singleton-test-file")
  val factorTestFile = asStr("mandolin.gm.factor-test-file")
  
  val subGradEpochs = asInt("mandolin.gm.subgrad-epochs")
  val sgAlpha   = asDouble("mandolin.gm.sg-alpha")
}
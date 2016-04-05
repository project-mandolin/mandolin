package org.mitre.mandolin.gm

import org.mitre.mandolin.glp.GLPModelSettings

class FactorGraphSettings(args: Array[String]) extends GLPModelSettings(args) {
  
  val singletonFile = asStr("mandolin.gm.singleton-file")
  val factorFile = asStr("mandolin.gm.factor-file")
  
  
}
package org.mitre.mandolin.gm

import org.mitre.mandolin.mlp.MandolinMLPSettings
import net.ceedubs.ficus.Ficus._

class FactorGraphSettings(args: Array[String]) extends MandolinMLPSettings(args) {

  val singletonFile = asStr("mandolin.gm.singleton-file")
  val factorFile = asStrOpt("mandolin.gm.factor-file")

  val singletonTestFile = asStrOpt("mandolin.gm.singleton-test-file")
  val factorTestFile = asStrOpt("mandolin.gm.factor-test-file")

  val subGradEpochs = asInt("mandolin.gm.subgrad-epochs")
  val sgAlpha = asDouble("mandolin.gm.sg-alpha")
  val inferAlgorithm = asStrOpt("mandolin.gm.infer-algorithm")
  val isSparse = this.netspec.head("ltype").equals("InputSparse")
  
  val factorSpec = try {
    config.as[List[Map[String, String]]]("mandolin.gm.factor-spec")
  } catch {
    case _: Throwable =>
      Nil
  }
  
  val factorSparse = factorSpec.head("ltype").equals("InputSparse")
}
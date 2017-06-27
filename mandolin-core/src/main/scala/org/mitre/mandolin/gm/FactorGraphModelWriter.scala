package org.mitre.mandolin.gm

import org.mitre.mandolin.mlp.{ ANNetwork, MMLPWeights }
import org.mitre.mandolin.util.{ Alphabet, IOAssistant }

case class FactorGraphModelSpec(swts: MMLPWeights, fwts: MMLPWeights, snet: ANNetwork, fnet: ANNetwork,
                                sla: Alphabet, fla: Alphabet, sfa: Alphabet, ffa: Alphabet)

abstract class FactorGraphModelWriter {

  def writeModel(io: IOAssistant, filePath: String,
                 sw: MMLPWeights, sa: Alphabet, sla: Alphabet, sann: ANNetwork,
                 fw: MMLPWeights, fa: Alphabet, fla: Alphabet, fann: ANNetwork) : Unit
  
}

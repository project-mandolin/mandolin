package org.mitre.mandolin.gm

import org.mitre.mandolin.glp.{ ANNetwork, GLPWeights }
import org.mitre.mandolin.util.{ Alphabet, IOAssistant }

case class FactorGraphModelSpec(swts: GLPWeights, fwts: GLPWeights, snet: ANNetwork, fnet: ANNetwork,
    sla: Alphabet, fla: Alphabet, sfa: Alphabet, ffa: Alphabet)

abstract class FactorGraphModelWriter {

  def writeModel(io: IOAssistant, filePath: String, 
      sw: GLPWeights, sa: Alphabet, sla: Alphabet, sann: ANNetwork,
      fw: GLPWeights, fa: Alphabet, fla: Alphabet, fann: ANNetwork) : Unit
      
  
}
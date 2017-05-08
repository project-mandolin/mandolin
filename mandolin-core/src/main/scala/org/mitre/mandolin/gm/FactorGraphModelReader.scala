package org.mitre.mandolin.gm

import org.mitre.mandolin.util.IOAssistant

abstract class FactorGraphModelReader {
  
  def readModel(io: IOAssistant, filePath: String) : FactorGraphModelSpec

}
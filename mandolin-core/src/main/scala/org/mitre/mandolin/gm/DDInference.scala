package org.mitre.mandolin.gm

abstract class DDInference {
  def mapInfer(f: Vector[MultiFactor], s: Vector[SingletonFactor], maxN: Int)
}


class SubgradientInference(val fm: FactorModel, val sm: FactorModel) extends DDInference {
  
  var alpha = 0.1f
  var decay = 0.99f
  
  def mapInfer(f: Vector[MultiFactor], s: Vector[SingletonFactor], maxN: Int) = {
    var i = 0; while (i < maxN) {
      alpha *= decay
      f foreach {factor => factor.setMode(fm, true)}
      s foreach {single => single.setMode(sm, true)}
      f foreach {factor =>
        val fAssignment = factor.varAssignment
        var j = 0; while (j < factor.numVars) {
          val fjAssign = fAssignment(j)
          val sjAssign = factor.singletons(j).currentAssignment
          if (fjAssign != sjAssign) { // disagreement in variable assignment
            factor.deltas(j)(fjAssign) += alpha
            factor.deltas(j)(sjAssign) -= alpha
          }
          j += 1
        }
        }
      i += 1
    }
    s foreach {single => single.setMode(sm, true)}    
  }
}
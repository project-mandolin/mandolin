package org.mitre.mandolin.gm

abstract class DDInference {
  def mapInfer(f: Vector[MultiFactor], s: Vector[SingletonFactor], maxN: Int)
}


/**
 * Follows simple subgradient approach outlined in Sontag et al. 2011
 */
class SubgradientInference(val fm: FactorModel, val sm: FactorModel, init: Double) extends DDInference {
  
  var alpha = init.toFloat
  
  def mapInfer(f: Vector[MultiFactor], s: Vector[SingletonFactor], maxN: Int) = {
    var i = 0; while (i < maxN) {
      alpha /= (1.0f + i)
      f foreach {factor => factor.setModeHard(fm, true)}
      s foreach {single => single.setModeHard(sm, true)}
      f foreach {factor =>
        val fAssignment = factor.varAssignment
        print("Factor mode assignment: ")
        fAssignment foreach {v => print(" " + v)}
        println
        var j = 0; while (j < factor.numVars) {
          val fjAssign = fAssignment(j)
          val sjAssign = factor.singletons(j).currentAssignment
          if (fjAssign != sjAssign) { // disagreement in variable assignment
            println("Disagreement on var " + j + " true = " + factor.singletons(j).label + " factor = " + fjAssign + " single = " + sjAssign)
            factor.deltas(j)(fjAssign) += alpha
            factor.deltas(j)(sjAssign) -= alpha
          }
          j += 1
        }
      }
      i += 1
    }
    // simple MAP primal solution:
    s foreach {single => single.setModeHard(sm, true)}    
  }
}

trait ComputeFullGradient {
  def computeGradientNorm(factors: Vector[MultiFactor], singletons: Vector[SingletonFactor]) = {
    var n = 0.0
    singletons foreach {s =>
      var iVal = 0; while (iVal < s.varOrder) {
        val uvMarg = s.reparameterizedMarginals(iVal)
        s.parentFactors foreach {case (c, vInd) =>
          val diff = (uvMarg - c.fixedVarMarginals(vInd)(iVal))
          n += (diff * diff)
        }
        iVal += 1
      }
    }
    println("Grad norm: " + math.sqrt(n))
  }
  
  
}

/**
 * Uses approach in Meshi et al. "Convergence Rate Analysis of MAP Coordinate Minimization Algorithms"
 * Smoothed dual MAP problem using soft-max 
 */
class SmoothedGradientInference(val fm: FactorModel, val sm: FactorModel, init: Double) extends DDInference with ComputeFullGradient {
  
  var alpha = init.toFloat
  
  def mapInfer(factors: Vector[MultiFactor], singletons: Vector[SingletonFactor], maxN: Int) = {
    var i = 0; while (i < maxN) {
      //alpha /= (1.0f + i)
      singletons foreach {s =>
        s.getMode(sm, true, 10.0f)
        val reparams = s.reparameterizedMarginals
        print("Reparameterized marginal distribution (singleton): ")
        reparams foreach {v => print(" " + v)}
        println
        var iVal = 0; while (iVal < s.varOrder) {
          val mu_i = s.reparameterizedMarginals(iVal)
          s.parentFactors foreach { case (c, vInd) =>
            c.getMode(fm, true, 10.0f)
            val diff = mu_i - c.fixedVarMarginals(vInd)(iVal).toFloat
            println("diff => " + diff)
            c.deltas(vInd)(iVal) -= (alpha * (mu_i - c.fixedVarMarginals(vInd)(iVal))).toFloat 
          }
          iVal += 1
        }
      }
      computeGradientNorm(factors, singletons)
      i += 1
    }
    // simple MAP primal solution:
    singletons foreach { single => single.setMode(sm, true) }
  }
}

class StarCoordinatedBlockMinimizationInference(val fm: FactorModel, val sm: FactorModel, tau: Double) extends DDInference with ComputeFullGradient {
  
  def mapInfer(factors: Vector[MultiFactor], singletons: Vector[SingletonFactor], maxN: Int) = {
    println("Performing MAP inference using star-updates")
    var i = 0; while (i < maxN) {
      val shuffled = util.Random.shuffle(singletons)
      shuffled foreach {s =>
        s.getMode(sm, true, tau)
        val parents = s.parentFactors
        var iVal = 0; while (iVal < s.varOrder) { // iterate over var domain                    
          val u_i = s.reparameterizedMarginals(iVal)
          var margSums = 0.0
          parents foreach {case (c,vInd) =>
            c.getMode(fm, true, tau)
            margSums += math.log(c.fixedVarMarginals(vInd)(iVal)) }
          margSums += math.log(u_i)
          margSums /= (1.0 + parents.length)
          val mSum = margSums.toFloat / tau.toFloat
          // once fixed margSums computed, perform local closed-form update to deltas
          parents foreach {case (c,vInd) =>
            // update deltas -- STAR update --
            c.deltas(vInd)(iVal) += ((math.log(c.fixedVarMarginals(vInd)(iVal)).toFloat / tau.toFloat) - mSum)
          }
          iVal += 1
        }
      }
      computeGradientNorm(factors, singletons)
      i += 1
    }
    // simple MAP primal solution:
    singletons foreach { single => single.setMode(sm, true) }
  }
  
}
package org.mitre.mandolin.gm

abstract class DDInference {
  def mapInfer(f: Vector[MultiFactor], s: Vector[SingletonFactor], maxN: Int)
}


/**
 * Follows simple subgradient approach outlined in Sontag et al. 2011
 */
class SubgradientInference(val fm: PairFactorModel, val sm: SingletonFactorModel, init: Double) extends DDInference {
  
  val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  var alpha = init.toFloat
  
  def mapInfer(f: Vector[MultiFactor], s: Vector[SingletonFactor], maxN: Int) = {
    logger.info("Inference using simple sub-gradient method")
    var i = 0; while (i < maxN) {
      alpha /= (1.0f + i)
      f foreach {factor => factor.setModeHard(fm, true)}
      s foreach {single => single.setModeHard(sm, true)}
      var adjustments = 0
      f foreach {factor =>
        val fAssignment = factor.varAssignment
        val sb = new StringBuilder
        fAssignment foreach {v => sb append (" " + v)}
        logger.info(sb.toString)
        var j = 0; while (j < factor.numVars) {
          val fjAssign = fAssignment(j)
          val sjAssign = factor.singletons(j).currentAssignment
          if (fjAssign != sjAssign) { // disagreement in variable assignment
            adjustments += 1
            logger.info("Disagreement on var " + j + " true = " + factor.singletons(j).label + " factor = " + fjAssign + " single = " + sjAssign)
            factor.deltas.get(j)(fjAssign) += alpha
            factor.deltas.get(j)(sjAssign) -= alpha
          }
          j += 1
        }
      }
      logger.info("Made " + adjustments + " adjustments on inference epoch " + i)
      i += 1
    }
    // simple MAP primal solution:
    s foreach {single => single.setModeHard(sm, true)}    
  }
}

trait ComputeFullGradient {
  val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  def computeGradientNorm(factors: Vector[MultiFactor], singletons: Vector[SingletonFactor]) = {
    var n = 0.0
    singletons foreach {s =>
      var iVal = 0; while (iVal < s.varOrder) {
        val uvMarg = s.reparameterizedMarginals(iVal)
        s.parentFactors foreach {case (c, vInd) =>
          val diff = (uvMarg - c.fixedVarMarginals.get(vInd)(iVal))
          n += (diff * diff)
        }
        iVal += 1
      }
    }
    logger.info("Grad norm: " + math.sqrt(n))
  }  
}

/**
 * Uses approach in Meshi et al. "Convergence Rate Analysis of MAP Coordinate Minimization Algorithms"
 * Smoothed dual MAP problem using soft-max 
 */
class SmoothedGradientInference(val fm: PairFactorModel, val sm: SingletonFactorModel, init: Double) extends DDInference with ComputeFullGradient {
  
  var alpha = init.toFloat
  
  def mapInfer(factors: Vector[MultiFactor], singletons: Vector[SingletonFactor], maxN: Int) = {
    var i = 0; while (i < maxN) {
      //alpha /= (1.0f + i)
      var singlesProcessed = 0
      var tt = System.nanoTime
      singletons foreach {s =>
        s.getMode(sm, true, 10.0f)
        singlesProcessed += 1
        if ((singlesProcessed % 1000) == 0) {
          val ct = System.nanoTime
          logger.info("Updates ["+singlesProcessed+"] in " + ((ct - tt)/1E9) + " seconds")
          tt = ct
        }
        var iVal = 0; while (iVal < s.varOrder) {
          val mu_i = s.reparameterizedMarginals(iVal)
          s.parentFactors foreach { case (c, vInd) =>
            c.getMode(fm, true, 10.0f)
            c.deltas.get(vInd)(iVal) -= (alpha * (mu_i - c.fixedVarMarginals.get(vInd)(iVal))).toFloat 
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

class StarCoordinatedBlockMinimizationInference(val fm: PairFactorModel, val sm: SingletonFactorModel, tau: Double) extends DDInference with ComputeFullGradient {
  
  
  def mapInfer(factors: Vector[MultiFactor], singletons: Vector[SingletonFactor], maxN: Int) = {
    logger.info("Performing MAP inference using star-updates")
    var i = 0; while (i < maxN) {
      val shuffled = util.Random.shuffle(singletons)
      var singlesProcessed = 0
      var tt = System.nanoTime
      shuffled foreach {s =>
        s.getMode(sm, true, tau)
        val parents = s.parentFactors
        singlesProcessed += 1
        if ((singlesProcessed % 1000) == 0) {
          val ct = System.nanoTime
          logger.info("Updates ["+singlesProcessed+"] in " + ((ct - tt)/1E9) + " seconds")
          tt = ct
        }
        var iVal = 0; while (iVal < s.varOrder) { // iterate over var domain                    
          val u_i = s.reparameterizedMarginals(iVal)
          var margSums = 0.0
          parents foreach {case (c,vInd) =>            
            c.getMode(fm, true, tau)     // can't cache this as re-parameterized marginals are updated via delta update below ...
            margSums += math.log(c.fixedVarMarginals.get(vInd)(iVal)) }
          margSums += math.log(u_i)   
          margSums /= (1.0 + parents.length)
          val mSum = margSums.toFloat / tau.toFloat
          // once fixed margSums computed, perform closed-form update to deltas
          parents foreach {case (c,vInd) =>
            // update deltas -- STAR update --  Update all parents
            c.deltas.get(vInd)(iVal) += ((math.log(c.fixedVarMarginals.get(vInd)(iVal)).toFloat / tau.toFloat) - mSum)
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
package org.mitre.mandolin.gm

import org.mitre.mandolin.glp.{GLPPredictor, ANNetwork, GLPWeights, GLPFactor}

abstract class Factor {
  def getMode(dual: Boolean) : Int
}


class SingletonFactor(val variable: CategoricalVariable, val fm: FactorModel, input: GLPFactor) extends Factor {
  
  // keeps a list of parent factors and the local index of THIS singleton factor
  // in the parent factors set of singletons
  var parentFactors : List[(MultiFactor,Int)] = Nil
  
  def getMode(dual: Boolean) : Int = {
    if (!dual) fm.getMode(input)
    else {
      val fmarg = fm.getFullMarginal(input)
      var bestSc = -Double.MaxValue
      var best = 0 // index into best assignment
      fmarg foreach {case (v,ind) =>
        var av = v
        parentFactors foreach {case (mfactor, lid) =>
          av += mfactor.deltas(lid)(ind)
          }
        if (av > bestSc) {
          bestSc = av
          best = ind
        }
      }
      best
    }
  }
}

class MultiFactor(val variableOrder: Int, val singletons: Array[SingletonFactor], val fm: FactorModel, input: GLPFactor) 
extends Factor {
  
  val numVars = singletons.length
  val tmpAssignment = Array.fill(numVars)(0.0f)
  val indexAssignmentMap = Array.tabulate(numConfigs)(indexToAssignment)
  
  // each singleton sub-factor is a row
  // each possible value for the corresponding variable is an entry in the row
  val deltas = Array.fill(numVars,variableOrder)(0.0f)
  
  /*
   * Number of possible configurations of variable assignments for this factor
   * Assume all variables have the same order (for now) - i.e. same state-space
   */
  def numConfigs = math.pow(variableOrder, numVars).toInt

  /*
   * Assignments are just base-N numbers where N is the variableOrder (i.e. size of domain)
   */
  def assignmentToIndex(assignment: Array[Int]) : Int = {
    var s = 0
    var i = 0; while (i < numVars) {
      val base = variableOrder * (i+1)
      s += assignment(i) * base
      i += 1
    }
    s
  }
  
  private def indexToAssignment(ind: Int) : Array[Int] = {
    // simple, but not super efficient probably - just compute once
    val ss = Integer.toString(ind, variableOrder)  
    Array.tabulate(numVars){i => ss.charAt(i) - 48}
  }
  
  def getFullMarginal = {
    val fmarg = fm.getFullMarginal(input)
    
  }
  
  def getConstrainedMarginal(constraints: Array[Option[Int]]) = {
    val fullMarginal = fm.getFullMarginal(input)
    var z = 0.0
    var i = 0; while(i < numVars) {
      
      i += 1
    }
  }
  
  def getMode(dual: Boolean) = {
    if (!dual) {
      fm.getMode(input)
    } else {
      val fmarg = fm.getFullMarginal(input)
      var bestSc = -Double.MaxValue
      var best = 0 // index into best assignment
      fmarg foreach {case (v,ind) =>
        val assignment = indexAssignmentMap(ind)
        var av = v
        var i = 0; while (i < numVars) {
          av -= deltas(i)(assignment(i))
          i += 1
        }
        if (av > bestSc) {
          bestSc = av
          best = ind
        }
      }
      best
    }
  }
  
}


/*
 * This is the parametric model that produces distributions over variable
 * configurations for a factor
 */
class FactorModel(glp: GLPPredictor, wts: GLPWeights) {
  
  def getFullMarginal(input: GLPFactor) = glp.getScoredPredictions(input, wts)
    
  def getMode(input: GLPFactor) = glp.getPrediction(input, wts)
  
}
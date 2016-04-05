package org.mitre.mandolin.gm

import org.mitre.mandolin.glp.{GLPPredictor, ANNetwork, GLPWeights, GLPFactor, StdGLPFactor}
import org.mitre.mandolin.util.{DenseTensor1 => DenseVec}

abstract class GMFactor {
  var currentAssignment : Int = 0
  def getMode(fm: FactorModel, dual: Boolean) : Int
   
  def getInput : GLPFactor
  def indexToAssignment(i: Int) : Array[Int]
  def assignmentToIndex(ar: Array[Int]) : Int
  
  def setMode(fm: FactorModel, dual: Boolean) : Unit = {
    currentAssignment = getMode(fm, dual)
  }
}


class SingletonFactor(input: GLPFactor, val label: Int) extends GMFactor {
  
  // keeps a list of parent factors and the local index of THIS singleton factor
  // in the parent factors set of singletons
  var parentFactors : List[(MultiFactor,Int)] = Nil
  
  def getInput = input
  
  def indexToAssignment(i: Int) = Array(i)
  def assignmentToIndex(ar: Array[Int]) = ar(0)
  
  def addParent(mf: MultiFactor, i: Int) = parentFactors = (mf,i) :: parentFactors 
  
  def getMode(fm: FactorModel, dual: Boolean) : Int = {
    if (!dual) fm.getMode(input)
    else {
      val fmarg = fm.getFullMarginal(input)
      var bestSc = -Double.MaxValue
      var best = 0 // index into best assignment
      fmarg foreach {case (v,ind) =>
        var av = v
        parentFactors foreach {case (mfactor, lid) =>
          //println("Adding parent factor deltas to singleton score: " + mfactor.name + " ("+lid+","+ind+") => " + mfactor.deltas(lid)(ind))
          av += mfactor.deltas(lid)(ind) }          
        if (av > bestSc) {
          bestSc = av
          best = ind
        }
      }
      //println("Singleton assignment = " + best + " with score = " + bestSc)
      best
    }
  }
  
}

class MultiFactor(val variableOrder: Int, val singletons: Array[SingletonFactor], dvec: DenseVec, val name: String) extends GMFactor {
  
  val numVars = singletons.length
  /*
   * Number of possible configurations of variable assignments for this factor
   * Assume all variables have the same order (for now) - i.e. same state-space
   */
  val numConfigs = math.pow(variableOrder, numVars).toInt
    
  val tmpAssignment = Array.fill(numVars)(0.0f)
  val bases = Vector.tabulate(numVars){i => math.pow(variableOrder,i).toInt}
  
  // maps single index values for assignments to array of local variable assignment 
  val indexAssignmentMap = Array.tabulate(numConfigs)(indexToAssignment)
  
  // each singleton sub-factor is a row
  // each possible value for the corresponding variable is an entry in the row
  val deltas = Array.fill(numVars,variableOrder)(0.0f)
  
  def varAssignment : Array[Int] = indexAssignmentMap(currentAssignment)
  
  val input = {
    val lv = DenseVec.zeros(numConfigs)
    val assignment = singletons map {s => s.label}
    val l_ind = assignmentToIndex(assignment)
    lv.update(l_ind,1.0f) // one-hot encoding
    new StdGLPFactor(-1, dvec, lv, None)     
  }
  
  def getInput = input

  /*
   * Assignments are just base-N numbers reversed where N is the variableOrder (i.e. size of domain)
   */
  def assignmentToIndex(assignment: Array[Int]) : Int = {
    var s = 0
    var i = 0; while (i < numVars) {      
      s += assignment(i) * bases(i)
      i += 1
    }
    s
  }
    
  def indexToAssignment(ind: Int) : Array[Int] = {
    val ar = Array.fill(numVars)(0)
    var i = numVars - 1
    var nn = ind
    while (i >= 0) {
      val base = bases(i)
      ar(i) = nn / base
      nn -= (base * ar(i))
      i -= 1
    }
    ar
  }
  
  def getMode(fm: FactorModel, dual: Boolean) = {
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
class FactorModel(glp: GLPPredictor, val wts: GLPWeights) {
  
  def getFullMarginal(input: GLPFactor) = glp.getScoredPredictions(input, wts)
    
  def getMode(input: GLPFactor) = glp.getPrediction(input, wts)
  
}
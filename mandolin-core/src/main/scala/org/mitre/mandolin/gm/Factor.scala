package org.mitre.mandolin.gm

import org.mitre.mandolin.mlp.{CategoricalMMLPPredictor, ANNetwork, MMLPWeights, MMLPFactor, StdMMLPFactor}
import org.mitre.mandolin.util.{DenseTensor1 => DenseVec}

abstract class GMFactor {
  var currentAssignment : Int = 0
  def getMode(fm: FactorModel, dual: Boolean) : Int
  def getMode(fm: FactorModel, dual: Boolean, tau: Double) : Int
  def getModeHard(fm: FactorModel, dual: Boolean) : Int
   
  def getInput : MMLPFactor
  def indexToAssignment(i: Int) : Array[Int]
  def assignmentToIndex(ar: Array[Int]) : Int
  
  def setMode(fm: FactorModel, dual: Boolean) : Unit = {
    currentAssignment = getMode(fm, dual)
  }
  
  def setMode(fm: FactorModel, dual: Boolean, tau: Double) : Unit = {
    currentAssignment = getMode(fm, dual, tau)
  }
  
  def setModeHard(fm: FactorModel, dual: Boolean) : Unit = {
    currentAssignment = getModeHard(fm, dual)
  }
}


class SingletonFactor(input: MMLPFactor, val label: Int) extends GMFactor {
  
  // keeps a list of parent factors and the standalone index of THIS singleton factor
  // in the parent factors set of singletons
  var parentFactors : List[(MultiFactor,Int)] = Nil
  
  val varOrder = input.getOutput.getDim
  
  val reparameterizedMarginals = Array.fill(varOrder)(0.0)
  
  def getInput = input
  
  def indexToAssignment(i: Int) = Array(i)
  def assignmentToIndex(ar: Array[Int]) = ar(0)
  
  def addParent(mf: MultiFactor, i: Int) = parentFactors = (mf,i) :: parentFactors 
  
  def getMode(fm: FactorModel, dual: Boolean) = getMode(fm, dual, 10.0)
  
  def getMode(fm: FactorModel, dual: Boolean, tau: Double) : Int = {
    if (!dual) fm.getMode(input)
    else {
      val fmarg = fm.getFullMarginal(input)
      
      var bestSc = -Double.MaxValue
      var best = 0 // index into best assignment
      var zSum = 0.0
      fmarg foreach {case (v,ind) =>
        var av = v
        parentFactors foreach { case (mfactor, lid) =>
          //println("Adding parent factor deltas to singleton score: " + mfactor.name + " ("+lid+","+ind+") => " + mfactor.deltas(lid)(ind))
          av += mfactor.deltas(lid)(ind) }
        val ss = math.exp(av * tau)
        reparameterizedMarginals(ind) = ss
        zSum += ss
        if (av > bestSc) {
          bestSc = av
          best = ind
        }
      }
      var i = 0; while (i < varOrder) {
        reparameterizedMarginals(i) /= zSum   // normalize
        i += 1
      }
      //println("Singleton assignment = " + best + " with score = " + bestSc)
      best
    }
  }
  
  def getModeHard(fm: FactorModel, dual: Boolean) : Int = {
    val fmarg = fm.getFullMarginal(input)
    var bestSc = -Double.MaxValue
    var best = 0 // index into best assignment
    fmarg foreach {case (v,ind) =>
      var av = v
      parentFactors foreach {case (mfactor, lid) => av += mfactor.deltas(lid)(ind) } // add in parent factor deltas
      if (av > bestSc) {
        bestSc = v
        best = ind
      }
    }
    best
  }
  
}

class MultiFactor(val indexAssignmentMap: Array[Array[Int]], 
    val variableOrder: Int, val singletons: Array[SingletonFactor], dvec: DenseVec, val name: String) extends GMFactor {
  
  val numVars = singletons.length
  /*
   * Number of possible configurations of variable assignments for this factor
   * Assume all variables have the same order (for now) - i.e. same state-space
   */
  val numConfigs = math.pow(variableOrder, numVars).toInt
    
  val tmpAssignment = Array.fill(numVars)(0.0f)
  val bases = Vector.tabulate(numVars){i => math.pow(variableOrder,i).toInt}
  
  
  // let this keep track of reparameterized marginals
  val reparamMarginals = Array.fill(numConfigs)(0.0)
  
  /*
   * This keeps track of single fixed variable marginalizations
   * fixedVarMarginals(i,v) is the marginalized probability of the factor with variable i fixed to v
   */
  val fixedVarMarginals = Array.fill(numVars,variableOrder)(0.0)
  
  // maps single index values for assignments to array of standalone variable assignment
  //val indexAssignmentMap = Array.tabulate(numConfigs)(indexToAssignment)
  
  // each singleton sub-factor is a row
  // each possible value for the corresponding variable is an entry in the row
  val deltas = Array.fill(numVars,variableOrder)(0.0f)
  
  def varAssignment : Array[Int] = indexAssignmentMap(currentAssignment)
  
  val input = {
    val lv = DenseVec.zeros(numConfigs)
    val assignment = singletons map {s => s.label}    
    val l_ind = assignmentToIndex(assignment)
    lv.update(l_ind,1.0f) // one-hot encoding
    new StdMMLPFactor(-1, dvec, lv, None)
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
  
    
  def indexToAssignment(i: Int) = indexAssignmentMap(i)
  
  def getMode(fm: FactorModel, dual: Boolean) = getMode(fm, dual, 10.0)
  
  def getMode(fm: FactorModel, dual: Boolean, tau: Double) = {
    if (!dual) {
      fm.getMode(input)
    } else {
      val fmarg = fm.getFullMarginal(input)
      var bestSc = -Double.MaxValue
      var best = 0 // index into best assignment
      var zSum = 0.0
      //println("Original marginals (factor " + name + "): ")
      //fmarg foreach { case (v,ind) => print(" "+ind+"=>"+v)}
      //println
      var i = 0; while (i < numVars) {
        var j = 0; while (j < variableOrder) {
          fixedVarMarginals(i)(j) = 0.0f
          j += 1 
        }
        i += 1
      }
      fmarg foreach {case (v,ind) =>
        val assignment = indexAssignmentMap(ind)
        var av = v 
        var i = 0; while (i < numVars) {
          av -= deltas(i)(assignment(i))
          i += 1
        }
        val ss = math.exp(av * tau)
        reparamMarginals(ind) = ss  // Store reparameterized marginals
        zSum += ss
        i = 0; while (i < numVars) {
          fixedVarMarginals(i)(assignment(i)) += ss
          i += 1
        }
        if (av > bestSc) {
          bestSc = av
          best = ind
        }
      }
      /*
      println("Reparameterized marginals factor: ")
      i = 0; while (i < numConfigs) {
        reparamMarginals(i) /= zSum
        print(" " + reparamMarginals(i))
        i += 1
      }
      println
      */
      // normalize single-fixed variable marginalizations
      /*
      println("Un-normalized Factor marginalizations\n--------")
      i = 0; while (i < numVars) {
        var j = 0; while (j < variableOrder) {
          println("P(x_"+i+"="+j+"|.)="+fixedVarMarginals(i)(j))
          j += 1
        }
        i += 1
      }
      * 
      */
      // normalize single-fixed variable marginalizations
      //println("Factor marginalizations\n--------")
      i = 0; while (i < numVars) {
        var j = 0; while (j < variableOrder) {
          fixedVarMarginals(i)(j) /= zSum
          //println("P(x_"+i+"="+j+"|.)="+fixedVarMarginals(i)(j))
          j += 1
        }
        i += 1
      }
            
      best
    }
  }  
  
  def getModeHard(fm: FactorModel, dual: Boolean) = {
    val fmarg = fm.getFullMarginal(input)
    var bestSc = -Double.MaxValue
    var best = 0 // index into best assignment
    fmarg foreach {case (v,ind) =>
      val assignment = indexAssignmentMap(ind)
      var av = v
      print("Factor pre-marginal: ")
      assignment foreach {v => print(" " + v)}
      println(" ==> " + v)
      var i = 0; while (i < numVars) {
        av -= deltas(i)(assignment(i))
        i += 1
      }
      print("Factor post/dual-marginal: ")
      assignment foreach {v => print(" " + v)}
      println(" ==> " + av)
      if (av > bestSc) {
        bestSc = av
        best = ind
      }
    }
    best
  }
  
}


/*
 * This is the parametric model that produces distributions over variable
 * configurations for a factor
 */
class FactorModel(glp: CategoricalMMLPPredictor, val wts: MMLPWeights) {
  
  def getFullMarginal(input: MMLPFactor) = glp.getScoredPredictions(input, wts)
    
  def getMode(input: MMLPFactor) = glp.getPrediction(input, wts)
  
}
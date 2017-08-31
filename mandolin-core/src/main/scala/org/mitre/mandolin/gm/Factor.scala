package org.mitre.mandolin.gm

import org.mitre.mandolin.mlp._
import org.mitre.mandolin.util.{DenseTensor1 => DenseVec, SparseTensor1 => SparseVec, Tensor1=>Vec}

abstract class GMFactor[F <: GMFactor[F]] {
  var currentAssignment : Int = 0
  def getMode(fm: FactorModel[F], dual: Boolean) : Int
  def getMode(fm: FactorModel[F], dual: Boolean, tau: Double) : Int
  def getModeHard(fm: FactorModel[F], dual: Boolean) : Int
   
  def getInput : MMLPFactor
  def indexToAssignment(i: Int) : Array[Int]
  def assignmentToIndex(ar: Array[Int]) : Int
  
  def setMode(fm: FactorModel[F], dual: Boolean) : Unit = {
    currentAssignment = getMode(fm, dual)
  }
  
  def setMode(fm: FactorModel[F], dual: Boolean, tau: Double) : Unit = {
    currentAssignment = getMode(fm, dual, tau)
  }
  
  def setModeHard(fm: FactorModel[F], dual: Boolean) : Unit = {
    currentAssignment = getModeHard(fm, dual)
  }
}


class SingletonFactor(input: MMLPFactor, val label: Int) extends GMFactor[SingletonFactor] {
  // keeps a list of parent factors and the standalone index of THIS singleton factor
  // in the parent factors set of singletons
  var parentFactors : List[(MultiFactor,Int)] = Nil
  
  val varOrder = input.getOutput.getDim
  
  val reparameterizedMarginals = Array.fill(varOrder)(0.0)
  
  def getInput = input
  
  def indexToAssignment(i: Int) = Array(i)
  def assignmentToIndex(ar: Array[Int]) = ar(0)
  
  def addParent(mf: MultiFactor, i: Int) = parentFactors = (mf,i) :: parentFactors 
  
  def getMode(fm: FactorModel[SingletonFactor], dual: Boolean) : Int = getMode(fm, dual, 10.0)
  
  def getMode(fm: FactorModel[SingletonFactor], dual: Boolean, tau: Double) : Int = {
    if (!dual) fm.getMode(this)
    else {
      val fmarg = fm.getFullMarginal(this)
      // println("Singleton original marginal: " + fmarg)
      var bestSc = -Double.MaxValue
      var best = 0 // index into best assignment
      var zSum = 0.0
      fmarg foreach {case (v,ind) =>
        var av = 0.0 // v
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
      // println("Singleton assignment = " + best + " with score = " + bestSc)
      best
    }
  }
  
  def getModeHard(fm: FactorModel[SingletonFactor], dual: Boolean) : Int = {
    val fmarg = fm.getFullMarginal(this)
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
    val variableOrder: Int, val singletons: Array[SingletonFactor], 
    dvec: Vec, val name: String) extends GMFactor[MultiFactor] {
  
  val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  
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
    dvec match {
      case vec: DenseVec => new StdMMLPFactor(-1, vec, lv, None)
      case vec: SparseVec => new SparseMMLPFactor(-1, vec, lv, None)
    }
         
  }
  
  def marginalInference(singleGlp: ANNetwork, pairGlp: ANNetwork, weights: MultiFactorWeights) : (Array[Float], Array[Float], Array[Float]) = {
    val pairUnit = getInput
    val singleUnit1 = singletons(0).getInput
    val singleUnit2 = singletons(1).getInput
    // logger.info("Pair unit FV instance = " + pairUnit + " # " + pairUnit.getUniqueKey)
    // logger.info("Singleton 1 = " + singleUnit1 + " # " + singleUnit1.getUniqueKey)
    // logger.info("Singleton 2 = " + singleUnit2 + " # " + singleUnit2.getUniqueKey)
    
    singleGlp.forwardPass(singleUnit1.getInput, singleUnit1.getOutput, weights.singleWeights)
    val f1Output = singleGlp.outLayer.getOutput(true).copy
    singleGlp.forwardPass(singleUnit2.getInput, singleUnit2.getOutput, weights.singleWeights)
    val f2Output = singleGlp.outLayer.getOutput(true).copy
    pairGlp.forwardPass(pairUnit.getInput, pairUnit.getOutput, weights.pairWeights)
    val pairOutput = pairGlp.outLayer.getOutput(true).copy.asArray
    val a1 = f1Output.asArray
    val a2 = f2Output.asArray    
    val dim = a1.length
    var sum = 0.0
    var maxScore = -Float.MaxValue
    // println("Input A1 = " + singleUnit1.getInput.asArray.toVector)
    // println("Input A2 = " + singleUnit2.getInput.asArray.toVector)
    val potentials = 
    Array.tabulate(dim){i =>
      Array.tabulate(dim){ j =>    
        // println("A1 of i = " + i + " = " + a1(i))
        // println("A2 of j = " + j + " = " + a2(j))
        val pairOut = pairOutput(assignmentToIndex(Array(i,j)))
        // println("Pair = " + pairOut)
        val sc = a1(i) + a2(j) + pairOut
        maxScore = math.max(maxScore, sc)
        sc
        }
      }    
    var i = 0; while (i < dim) {
      var j = 0; while (j < dim) {
        val potential = math.exp(potentials(i)(j) - maxScore)
        sum += potential
        potentials(i)(j) = potential.toFloat 
        j += 1
      }
      i += 1
    }
    i = 0; while (i < dim) {
      var j = 0; while (j < dim) {
        potentials(i)(j) /= sum.toFloat 
        j += 1
      }
      i += 1
    }
    
    val vec2 = Array.tabulate(dim){j =>
      var s = 0.0f
      var i = 0; while (i < dim) {
        s += potentials(i)(j)
        i += 1
      }
      s
    }

    val vec1 = Array.tabulate(dim){i =>
      var s = 0.0f
      val pi = potentials(i)
      var j = 0; while (j < dim) {
        s += pi(j)
        j += 1
      }
      s
    }
    
    // Re-run forward pass    
    singleGlp.forwardPass(singleUnit1.getInput, singleUnit1.getOutput, weights.singleWeights)
    val dt1 = new DenseVec(vec1)
    singleGlp.outLayer.setOutput(dt1.copy) // set this to the vec1 - but use a copy
    val gr1 = singleGlp.backpropGradients(singleUnit1.getInput, singleUnit1.getOutput, weights.singleWeights)

    singleGlp.forwardPass(singleUnit2.getInput, singleUnit2.getOutput, weights.singleWeights)
    val dt2 = new DenseVec(vec2)
    singleGlp.outLayer.setOutput(dt2.copy) // set this to the vec2

    val gr2 = singleGlp.backpropGradients(singleUnit2.getInput, singleUnit2.getOutput, weights.singleWeights)
    
    pairGlp.forwardPass(pairUnit.getInput, pairUnit.getOutput, weights.pairWeights)
    
    // flatten the potentials back to a single vector
    val pairVec = Array.tabulate(numConfigs){cInd =>
      val assignment = indexToAssignment(cInd)
      potentials(assignment(0))(assignment(1))
    }
    
    pairGlp.outLayer.setOutput(new DenseVec(pairVec).copy)
    val grPair = pairGlp.backpropGradients(pairUnit.getInput, pairUnit.getOutput, weights.pairWeights)

    gr1.addEquals(gr2, 1.0f)
    
    (pairVec, vec1, vec2)
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
  
  def getMode(fm: FactorModel[MultiFactor], dual: Boolean) : Int = getMode(fm, dual, 10.0)
  
  def getMode(fm: FactorModel[MultiFactor], dual: Boolean, tau: Double) : Int = {
    if (!dual) {
      fm.getMode(this)
    } else {
      val fmarg = fm.getFullMarginal(this)
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
  
  def getModeHard(fm: FactorModel[MultiFactor], dual: Boolean) = {
    val fmarg = fm.getFullMarginal(this)
    var bestSc = -Double.MaxValue
    var best = 0 // index into best assignment
    fmarg foreach {case (v,ind) =>
      val assignment = indexAssignmentMap(ind)
      var av = v
      // print("Factor pre-marginal: ")
      // assignment foreach {v => print(" " + v)}
      // println(" ==> " + v)
      var i = 0; while (i < numVars) {
        av -= deltas(i)(assignment(i))
        i += 1
      }
      // print("Factor post/dual-marginal: ")
      // assignment foreach {v => print(" " + v)}
      // println(" ==> " + av)
      if (av > bestSc) {
        bestSc = av
        best = ind
      }
    }
    best
  }
  
}


abstract class FactorModel[F <: GMFactor[F]] {
  def getFullMarginal(input: F) : Seq[(Float, Int)]
  def getMode(input: F) : Int
}

/*
 * This is the parametric model that produces distributions over variable
 * configurations for a factor
 */
class SingletonFactorModel(mmlp: CategoricalMMLPPredictor, val wts: MMLPWeights) extends FactorModel[SingletonFactor] {
  
  def getFullMarginal(input: SingletonFactor) = mmlp.getScoredPredictions(input.getInput, wts)
    
  def getMode(input: SingletonFactor) = mmlp.getPrediction(input.getInput, wts)
}

class PairFactorModel(pairGlp: ANNetwork, singleGlp: ANNetwork, val fullWts: MultiFactorWeights) extends FactorModel[MultiFactor] {
  
  def getFullMarginal(input: MultiFactor) : Seq[(Float, Int)] = {
    val (pair, v1, v2) = input.marginalInference(singleGlp, pairGlp, fullWts)
    IndexedSeq.tabulate(pair.length){i => (pair(i), i)}
  }
  
  def getMode(input: MultiFactor) = {
    val marg = getFullMarginal(input)
    var best = 0
    var bestSc = 0.0
    marg foreach {case (sc,i) => if (sc > bestSc) {best = i; bestSc = sc}}
    best
  }
}
package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.optimize.TrainingUnitEvaluator
import org.mitre.mandolin.predict.{TrainTester, EvalPredictor, RegressionConfusion, DiscreteConfusion}

import org.mitre.mandolin.util.{DenseTensor1 => DenseVec, SparseTensor1 => SparseVec, Tensor1 => Vec}


abstract class GLPFactor extends Serializable {
  def getId: Int
  def getInput : Vec
  def getOutput : DenseVec
  def getOneHot = { getOutput.argmax }
  def getUniqueKey : Option[String]
}

class StdGLPFactor(val ind: Int, in: DenseVec, out: DenseVec, val uniqueKey: Option[String]) extends GLPFactor with Serializable {
  def this(in: DenseVec, out: DenseVec) = this(0, in, out, None)
  def this(in: DenseVec, out: DenseVec, k: Option[String]) = this(0,in, out, k)
  override def toString() = {
    val sbuf = new StringBuilder
    for (i <- 0 until out.getDim) {
      sbuf append (" " + out(i)) 
    }
    sbuf.toString()
  }
  def getId = ind
  def getInput = in
  def getOutput = out
  def getUniqueKey = uniqueKey
}

class SparseGLPFactor(val ind: Int, in: SparseVec, out: DenseVec, val uniqueKey: Option[String]) extends GLPFactor with Serializable {
  def this(in: SparseVec, out: DenseVec, id: Option[String]) = this(0, in, out, id)
  def this(in: SparseVec, out: DenseVec) = this(0, in, out, None)
  def this(ind: Int, in: SparseVec, out: DenseVec) = this(ind, in, out, None)
  override def toString() = {
    val sbuf = new StringBuilder
    for (i <- 0 until out.getDim) {
      sbuf append (" " + out(i)) 
    }
    sbuf.toString()
  }
  def getId = ind
  def getInput = in
  def getOutput = out
  def getUniqueKey = uniqueKey
}

/**
 * @author wellner
 */
class GLPInstanceEvaluator(val glp: ANNetwork)
extends TrainingUnitEvaluator [GLPFactor, GLPWeights, GLPLossGradient] with Serializable {
      
  def evaluateTrainingUnit(unit: GLPFactor, weights: GLPWeights) : GLPLossGradient = { 
    val gr = glp.getGradient(unit.getInput, unit.getOutput, weights)
    new GLPLossGradient(glp.getCost, gr)
  }  
  
  def copy() = new GLPInstanceEvaluator(glp.sharedWeightCopy()) // this copy will share weights but have separate layer data members
}

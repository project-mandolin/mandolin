package org.mitre.mandolin.mlp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.optimize.{TrainingUnitEvaluator, Updater}
import org.mitre.mandolin.predict.{EvalPredictor, RegressionConfusion, DiscreteConfusion}

import org.mitre.mandolin.util.{DenseTensor1 => DenseVec, SparseTensor1 => SparseVec, Tensor1 => Vec}


abstract class MMLPFactor extends Serializable {
  def getId: Int
  def getInput : Vec
  def getOutput : Vec
  def getOneHot = { getOutput.argmax }
  def getUniqueKey : Option[String]
}

class StdMMLPFactor(val ind: Int, in: DenseVec, out: DenseVec, val uniqueKey: Option[String]) extends MMLPFactor with Serializable {
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

class SparseMMLPFactor(val ind: Int, in: SparseVec, out: Vec, val uniqueKey: Option[String]) extends MMLPFactor with Serializable {
  def this(in: SparseVec, out: Vec, id: Option[String]) = this(0, in, out, id)
  def this(in: SparseVec, out: Vec) = this(0, in, out, None)
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
class MMLPInstanceEvaluator[U <: Updater[MMLPWeights, MMLPLossGradient, U]](val glp: ANNetwork)
extends TrainingUnitEvaluator [MMLPFactor, MMLPWeights, MMLPLossGradient, U] with Serializable {

  def evaluateTrainingUnit(unit: MMLPFactor, weights: MMLPWeights, u: U) : MMLPLossGradient = {
    val gr = glp.getGradient(unit.getInput, unit.getOutput, weights)
    new MMLPLossGradient(glp.getCost, gr)
  }

  def copy() = new MMLPInstanceEvaluator(glp.sharedWeightCopy()) // this copy will share weights but have separate layer data members
}

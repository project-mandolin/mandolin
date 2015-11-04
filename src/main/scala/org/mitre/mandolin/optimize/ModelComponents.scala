package org.mitre.mandolin.optimize
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import scala.reflect.ClassTag
import org.mitre.mandolin.util.{ Tensor1, DenseTensor1, SparseTensor1 }

/**
 * Represents the weights (or parameters) for a particular model type.
 * Most linear models will just represent as an `Array[Double]`, but
 * other weights may consist of matrices (e.g. for multi-layer perceptrons).
 * 
 * @author Ben Wellner
 */ 
abstract class Weights[W <: Weights[W]](m: Double) extends Serializable {
  def this() = this(1.0)
  
  var mass = m
  
  def resetMass(v: Double = 1.0) = mass = v
  def weightAt(i: Int): Double
  val numWeights: Int

  def checkWeights() : Unit = throw new RuntimeException("Check weights not implemented")
  
  def asArray: Array[Double]
  def asTensor1: Tensor1

  def compose(otherWeights: W) : W
  def ++(otherWeights: W): W = add(otherWeights)
  def +=(otherWeights: W) : Unit = addEquals(otherWeights)
  def *=(v: Double): Unit = timesEquals(v)
  def l2norm : Double
  
  def add(otherWeights: W) : W
  def addEquals(otherWeights: W) : Unit
  def timesEquals(v: Double) : Unit

  def updateFromArray(ar: Array[Double]): W
  def copy() : W
  
  def compress() : Unit
  def decompress() : Unit
}

/**
 * Represents the loss and its gradient given one or more training units
 * and the current parameters.
 * @param loss - the loss value
 * @author Ben Wellner 
 */ 
abstract class LossGradient[LG <: LossGradient[LG]](val loss: Double) {
  def ++(other: LG): LG = add(other)
  def add(other: LG) : LG
  def asArray : Array[Double]
}

/**
 * Subclasses evaluate a single training unit of type `T`, returning a LossGradient.
 * @author Ben Wellner
 */ 
abstract class TrainingUnitEvaluator[T, W <: Weights[W], LG <: LossGradient[LG]] extends Serializable {
  
  /**
   * Primary method which returns a LossGradient for a given training unit with the current model parameters/weights, W.
   */ 
  def evaluateTrainingUnit(unit: T, modelParameters: W): LG
  def copy(): TrainingUnitEvaluator[T, W, LG]
}

/**
 * Updater subclasses update a set of model weights given a LossGradient.
 * @param updaterMass is the weight/mass assigned to this updater to take weighted averages
 * @author Ben Wellner
 */ 
abstract class Updater[W <: Weights[W], LG <: LossGradient[LG], U <: Updater[W,LG,U] ](var updaterMass: Double = 1.0) extends Serializable {
  /**
   * Modify the weights, W, given the provided LossGradient. Different ML algorithms can be realized with
   * different update methods.
   */ 
  def updateWeights(g: LG, w: W): Unit
  def resetLearningRates(v: Double): Unit
  def resetMass(v: Double = 1.0) : Unit = updaterMass = v 
  def copy() : U
  
  def getDiagnostic() : String = ""
  
  /**
   * Compose this updater with an updater `u`. Updaters may include dynamic learning
   * rates and these may need to be averaged or otherwise composed. 
   */ 
  def compose(u: U) : U
  
  /**
   * Override compress and decompress methods to provide effciency gains when sending
   * updaters over the wire.
   */ 
  def compress() : Unit = {}
  def decompress() : Unit = {}
}






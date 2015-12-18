package org.mitre.mandolin.predict.local
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import scala.reflect.ClassTag

import org.mitre.mandolin.optimize.{ Weights, LossGradient, Updater }
import org.mitre.mandolin.optimize.local.LocalOnlineOptimizer
import org.mitre.mandolin.transform.FeatureExtractor

import org.mitre.mandolin.optimize.local.LocalOptimizerEstimator

/**
 * Encapsulates training functionality, including feature extraction.
 * @param fe feature extractor
 * @param opt distributed online/batch optimizer (to minimize training loss)
 * @param persistLevel force Spark caching of results of feature extraction at specified level
 */ 
class LocalTrainer[IType, Un : ClassTag, W <: Weights[W]](
    val fe: FeatureExtractor[IType,Un],
    val opt: LocalOptimizerEstimator[Un, W]
    ) {
  
  /**
   * Extracts features and then trains a model with provided distributed optimizer
   * 
   * @param rdd input representation to trainer
   * @return weights and loss value (as a 2-tuple)
   */ 
  def trainWeights(rdd: Vector[IType]) : (W, Double) = {
    val _fe = fe
    val _opt = opt
    val fs = rdd map { _fe.extractFeatures }
    _opt.estimate(fs)
  }

  /**
   * Extracts features, mapping the input representation to a representation for computing loss/gradients
   * @param rdd input representation
   * @return rdd with training unit representation
   */ 
  def extractFeatures(rdd: Vector[IType]) : Vector[Un] = {
    val _fe = fe
    rdd map {v =>
      _fe.extractFeatures(v) }    
  }
  
  /**
   * Trains/updates weights based on one or more epochs of training. NOTE: relies on state preserved from previous
   * calls to this function held within the optimization object
   * @param rdd training unit rdd representation
   * @param number of training epochs
   * @return weights and loss value (as a 2-tuple)
   */ 
  def retrainWeights(rdd: Vector[Un], epochs: Int = 1) : (W, Double) = {
    val _opt = opt
    _opt.estimate(rdd, Some(epochs))
  }
}

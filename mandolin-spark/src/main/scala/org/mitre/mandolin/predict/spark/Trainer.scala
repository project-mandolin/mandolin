package org.mitre.mandolin.predict.spark
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import scala.reflect.ClassTag
import org.mitre.mandolin.optimize.{ Weights}
import org.mitre.mandolin.optimize.spark.DistributedOptimizerEstimator
import org.mitre.mandolin.transform.FeatureExtractor
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * Encapsulates training functionality, including feature extraction.
 * @param fe feature extractor
 * @param opt distributed online/batch optimizer (to minimize training loss)
 * @param persistLevel force Spark caching of results of feature extraction at specified level
 */ 
class Trainer[IType, U : ClassTag, W <: Weights[W]](
    val fe: FeatureExtractor[IType,U],
    val opt: DistributedOptimizerEstimator[U, W],
    persistLevel : StorageLevel = StorageLevel.MEMORY_ONLY
    ) {
  
  /**
   * Extracts features and then trains a model with provided distributed optimizer
   * 
   * @param rdd input representation to trainer
   * @return weights and loss value (as a 2-tuple)
   */ 
  def trainWeights(rdd: RDD[IType]) : (W, Double) = {
    val _fe = fe
    val _opt = opt
    val fs = rdd map { _fe.extractFeatures }
    fs.persist(persistLevel) // persist before estimation
    rdd.unpersist() // drop the RDD here once we have the results of feature extraction
    _opt.estimate(fs)
  }

  /**
   * Extracts features, mapping the input representation to a representation for computing loss/gradients
   * @param rdd input representation
   * @return rdd with training unit representation
   */ 
  def extractFeatures(rdd: RDD[IType]) : RDD[U] = {
    val _fe = fe
    rdd map { _fe.extractFeatures }    
  }
  
  /**
   * Trains/updates weights based on one or more epochs of training. NOTE: relies on state preserved from previous
   * calls to this function held within the optimization object
   * @param rdd training unit rdd representation
   * @param number of training epochs
   * @return weights and loss value (as a 2-tuple)
   */ 
  def retrainWeights(rdd: RDD[U], epochs: Int = 1) : (W, Double) = {
    val _opt = opt
    _opt.estimate(rdd, Some(epochs))
  }
}

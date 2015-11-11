package org.mitre.mandolin.optimize.spark
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */


import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import org.mitre.mandolin.util._

import org.mitre.mandolin.optimize.{Weights, LossGradient, BatchEvaluator, GenData, BatchOptimizer, Numerical, Result, Params}
import org.mitre.mandolin.optimize.ErrorCodes._


class RDDData[T](val rdd: RDD[T]) extends GenData[T]





trait DistributedOptimizerEstimator[T, W <: Weights[W]] {
  def estimate(data: RDD[T], mxEpochs: Option[Int] = None): (W, Double)
}



/**
 * Abstract distributed convex optimizer
 * @param dim  - number of dimensions/parameters in function to optimize
 */
trait DistributedConvexOptimizer[T] extends Numerical {
  var result: Option[Result] = None
  case class IterationData(var alpha: Double, val s: Array[Double], val y: Array[Double], var ys: Double)
}


/**
 * Limited Memory L-BFGS implementation with distributed evaluation of cost/loss and gradient
 * @param _dim dimensionality of parameter space
 * @param _weights model weights
 * @param _batchEvaluator distributed evaluator of loss function and gradient
 * @param _params L-BFGS parameters/settings
 * @author wellner 
 */
class DistributedLbfgsOptimizer[T, W <: Weights[W], G <: LossGradient[G]](_dim: Int, _weights: W, _batchEvaluator: BatchEvaluator[T, W, G], 
    _params: Params) 
  extends BatchOptimizer[T, W, G](_dim, _weights, _batchEvaluator, _params) 
  with DistributedOptimizerEstimator[T, W] {
      
  def estimate(data: RDD[T], mxEpochs: Option[Int] = None) : (W, Double) = estimate(new RDDData(data), mxEpochs)
  
}


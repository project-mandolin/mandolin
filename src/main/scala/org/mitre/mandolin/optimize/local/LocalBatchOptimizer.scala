package org.mitre.mandolin.optimize.local
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.optimize.spark.BatchOptimizer
import org.mitre.mandolin.optimize.{Weights, LossGradient, BatchEvaluator, GenData}
import org.mitre.mandolin.optimize.spark.{Params}

class VectorData[T](val vec: Vector[T]) extends GenData[T]

class LocalBatchOptimizer[T, W <: Weights[W], G <: LossGradient[G]](_dim: Int, _weights: W, _batchEvaluator: BatchEvaluator[T, W, G], 
    _params: Params) extends BatchOptimizer[T, W, G] (_dim, _weights, _batchEvaluator, _params) with LocalOptimizerEstimator[T, W] {

  def estimate(data: Vector[T], mxEpochs: Option[Int] = None) : (W, Double) = {
    estimate(new VectorData(data), mxEpochs)
  }    
}
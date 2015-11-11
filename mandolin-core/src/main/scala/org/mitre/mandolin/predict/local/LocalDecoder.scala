package org.mitre.mandolin.predict.local
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.optimize.Weights
import scala.reflect.ClassTag
import org.mitre.mandolin.transform.{ FeatureExtractor, LineProcessor }
import org.mitre.mandolin.predict.{Predictor, OutputConstructor, EvalPredictor, Confusion, ConfusionMatrix}


class LocalDecoder[IType, U : ClassTag, W <: Weights[W], R : ClassTag, OType : ClassTag](
    fe: FeatureExtractor[IType,U],
    val pr: Predictor[U, W, R],
    val oc: OutputConstructor[IType, R, OType]) {
  
  def run(inputs: Vector[IType], w: W) : Vector[OType] = {
    val extractFeatures = fe.extractFeatures _
    val predict         = 
      {u : U =>
        pr.getPrediction(u, w)
      }
    val constructOutput = oc.constructOutput _
    
    val punits = inputs map { i => (i,extractFeatures(i)) }
    val responses = punits map { case (i,u) => (i,u,predict(u)) }
    val outputs = responses map {case (i,u,r) => constructOutput(i,r,u.toString)}
    outputs.toVector
  }
}

/*
 * Output constructor should return an input type as well as the corresponding training/test factor
 * so that the correct output label can be generated.
 */
class LocalPosteriorDecoder[IType, U : ClassTag, W <: Weights[W], R : ClassTag, OType : ClassTag](
    fe: FeatureExtractor[IType,U],
    val pr: Predictor[U, W, R],
    val oc: OutputConstructor[IType, Seq[(Double,R)], OType]
    ) {
  
  def run(inputs: Vector[IType], w: W) : Vector[(OType, U)] = {
    val extractFeatures = fe.extractFeatures _
    val predict         = 
      {u : U =>
        pr.getScoredPredictions(u, w)
      }
    val constructOutput = oc.constructOutput _    
    val punits    = inputs map { i => (i, extractFeatures(i)) }
    val responses = punits map { case (i,u) => (i,u,predict(u)) }
    val outputs = responses map {case (i,u,r) => (constructOutput(i,r, u.toString), u)}
    outputs.toVector
  }
}

/**
 * A decoder the takes and eval predictor to generate results/scores for predictions on a test set
 * @param fe feature extractor
 * @param pr predictor able to evaluate output against a gold-standard
 * @tparam IType input representation
 * @tparam U training/testing unit representation
 * @tparam W model weights/parameter representation
 * @tparam R model response representation
 * @tparam C confusion matrix and/or prediction,ground-truth pairs for evaluation/scoring/analysis  
 */ 
class LocalEvalDecoder[IType, U : ClassTag, W <: Weights[W], R : ClassTag, C <: Confusion[C] : ClassTag](
    fe: FeatureExtractor[IType,U],
    pr: EvalPredictor[U, W, R, C]
    ) {
  
  def extractFeatures(inputs: Vector[IType]) : Vector[U] = {
    inputs map fe.extractFeatures    
  }
  
  /**
   * Evaluate test inputs by applying predictor, aggregating results and generating final confusion matrix
   * @param inputs rdd of inputs
   * @param wBc Spark broadcast containing the model weights
   * @return a confusion object for computing accuracy/f-measure/ROC curves, etc.
   */ 
  def eval(inputs: Vector[IType], w: W) : ConfusionMatrix = {
    val extractFeatures = fe.extractFeatures _
    val getConf         = 
      {u : U =>
        pr.getConfusion(u, w)
      }
    val punits = inputs map { extractFeatures }
    val finalConfusion = punits map { getConf } reduce {_ compose _ }
    finalConfusion.getMatrix
  }
  
  /**
   * Evaluate test inputs by applying predictor, aggregating results and generating final confusion matrix
   * @param inputs rdd of inputs
   * @param wBc Spark broadcast containing the model weights
   * @return a confusion object for computing accuracy/f-measure/ROC curves, etc.
   */ 
  def evalToConfusion(inputs: Vector[IType], w: W) : C = {
    val extractFeatures = fe.extractFeatures _
    val getConf         = 
      {u : U =>        
        pr.getConfusion(u, w)
      }
    val punits = inputs map { extractFeatures }
    punits map { getConf } reduce {_ compose _ }    
  }
  
  /**
   * Evaluate training units (output of feature extractor)
   * @param units rdd of training/testing units
   * @param wBc broadcast of weights
   * @return resulting confusion object
   */ 
  def evalUnits(units: Vector[U], w: W) : C = {
    val getConf = 
      {u : U =>
        val r = pr.getConfusion(u, w)
        r
      }
    val confusions = units map { getConf }
    val finalConfusion = units map { getConf } reduce {_ compose _ }
    finalConfusion        
  }
  
}

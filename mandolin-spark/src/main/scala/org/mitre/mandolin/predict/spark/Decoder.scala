package org.mitre.mandolin.predict.spark
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.mitre.mandolin.optimize.Weights
import scala.reflect.ClassTag
import org.mitre.mandolin.transform.{ FeatureExtractor, LineProcessor }
import org.mitre.mandolin.predict.{Predictor, OutputConstructor, EvalPredictor, Confusion, ConfusionMatrix}


abstract class AbstractDecoder[IType, U : ClassTag, W <: Weights[W], R : ClassTag](
    val fe: FeatureExtractor[IType, U]) {
  /**
   * Run feature extractor on test inputs
   * @param inputs rdd of input objects
   * @return rdd of training/test unit objects
   */ 
  def extractFeatures(inputs: RDD[IType]) : RDD[U] = {
    val _fe = fe
    inputs map { _fe.extractFeatures }    
  }
}

/**
 * Abstract decoder which maps `INPUT`s to `UNIT`s via provided `FeatureExtractor`.
 * The Predictor object maps `UNIT` objects to `RESPONSE` objects. `OUTPUT`
 * objects are created by applying the OutputConstructor to an `INPUT` and the corresponding
 * predicted `RESPONSE`.
 * @param fe maps `INPUT` to `UNIT`s
 * @param pr maps `UNIT` objects to `RESPONSE` objects
 * @param oc maps `INPUT`, `RESPONSE` pairs to `OUTPUT` objects
 * @tparam IType input representation (prior to feature extraction)
 * @tparam U training unit representation
 * @tparam W weight representation
 * @tparam R the response type obtained by applying the predictor `pr` to a (training/test) unit
 * @tparam OType the output type to produce given the response
 * @author Ben Wellner 
 */ 
class Decoder[IType, U : ClassTag, W <: Weights[W], R : ClassTag, OType : ClassTag](
    _fe: FeatureExtractor[IType,U],
    val pr: Predictor[U, W, R],
    val oc: OutputConstructor[IType, R, OType]
    ) extends AbstractDecoder[IType, U, W, R](_fe) {
  
  def run(inputs: RDD[IType], wBc: Broadcast[W]) : RDD[OType] = {
    // These functions redefined locally so that small closures are created and serialized within Spark
    val l_fe = fe
    val l_pr = pr
    val l_oc = oc
    val extractFeatures = l_fe.extractFeatures _
    val predict         = 
      {u : U =>
        val w = wBc.value
        l_pr.getPrediction(u, w)
      }
    val constructOutput = l_oc.constructOutput _
    // end of closure construction
    
    val punits = inputs map { i => (i,extractFeatures(i)) }
    val responses = punits map { case (i,u) => (i,u,predict(u)) }
    responses map {case (i,u,r) => constructOutput(i,r,u.toString)}
  }
}

class PosteriorDecoder[IType, U : ClassTag, W <: Weights[W], R : ClassTag, OType : ClassTag](
    _fe: FeatureExtractor[IType,U],
    val pr: Predictor[U, W, R],
    val oc: OutputConstructor[IType, Seq[(Float,R)], OType]
    ) extends AbstractDecoder[IType, U, W, R](_fe) {
  
  def run(inputs: RDD[IType], wBc: Broadcast[W]) : RDD[(OType, U)] = {
    // These functions redefined locally so that small closures are created and serialized for use within Spark
    val l_fe = fe
    val l_pr = pr
    val l_oc = oc
    val extractFeatures = l_fe.extractFeatures _
    val predict         = 
      {u : U =>
        val w = wBc.value
        l_pr.getScoredPredictions(u, w)
      }
    val constructOutput = l_oc.constructOutput _
    // end of closure construction
    
    val punits    = inputs map { i => (i, extractFeatures(i)) }
    val responses = punits map { case (i,u) => (i,u,predict(u)) }
    responses map {case (i,u,r) => (constructOutput(i,r, u.toString), u)}
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
class EvalDecoder[IType, U : ClassTag, W <: Weights[W], R : ClassTag, C <: Confusion[C] : ClassTag](
    _fe: FeatureExtractor[IType,U],
    pr: EvalPredictor[U, W, R, C]
    ) extends AbstractDecoder[IType, U, W, R](_fe) {
  
  /**
   * Evaluate test inputs by applying predictor, aggregating results and generating final confusion matrix
   * @param inputs rdd of inputs
   * @param wBc Spark broadcast containing the model weights
   * @return a confusion object for computing accuracy/f-measure/ROC curves, etc.
   */ 
  def eval(inputs: RDD[IType], wBc: Broadcast[W]) : ConfusionMatrix = {
    val _fe = fe
    val _wBc = wBc
    val ppr = pr
    val extractFeatures = _fe.extractFeatures _
    val getConf         = 
      {u : U =>
        val w = _wBc.value
        ppr.getConfusion(u, w)
      }
    val punits = inputs map { extractFeatures }
    val finalConfusion = punits map { getConf } reduce {_ compose _ }
    finalConfusion.getMatrix
  }

  /**
   * Evaluate training units (output of feature extractor)
   * @param units rdd of training/testing units
   * @param wBc broadcast of weights
   * @return resulting confusion object
   */ 
  def evalUnits(units: RDD[U], wBc: Broadcast[W]) : C = {
    val _wBc = wBc
    val ppr = pr
    val getConf = 
      {u : U =>
        val w = _wBc.value
        val r = ppr.getConfusion(u, w)
        r
      }
    val finalConfusion = units map { getConf } reduce {_ compose _ }
    finalConfusion        
  }
  
}

/**
 * Decoder extension for reading input to and from files.
 * @param sc Sparkcontext required to create initial RDDs
 * @param lp Lineprocessor for mapping file input strings to input types and for generating output strings from output types
 * @param decoder which maps input types to output types using a model 
 * @author Ben Wellner
 */ 
class FileBasedDecoder[IType : ClassTag, U, W <: Weights[W], R, OType](
    sc: SparkContext,
    lp: LineProcessor[IType, OType],
    decoder: Decoder[IType, U, W, R, OType]) {
  
  def decodeFile(ifile: String, ofile: String, numPartitions: Int, wBc: Broadcast[W]) : Unit = {
    val os = new java.io.PrintWriter(new java.io.FileOutputStream(new java.io.File(ofile)))
    val _lp      = lp
    val _decoder = decoder
    val lines        = sc.textFile(ifile, numPartitions)
    val lineToInput  = _lp.lineToInput _
    val outputToLine = _lp.outputToLine _
    val inputs       = lines map lineToInput
    val outputs      = _decoder.run(inputs, wBc)
    outputs.toLocalIterator foreach {o => os.write(outputToLine(o)); os.write('\n')}
    os.close()
  }
}

/**
 * @param _fe FeatureExtractor - maps <i>INPUTs</i> to <i>UNITs</i>
 * @param _pr Predictor - maps <i>UNIT</i> objects to <i>RESPONSE</i> objects
 * @param _oc OutputConstructor - maps <i>INPUT, RESPONSE</i> pairs to <i>OUTPUT</i> objects
 * @param sc `SparkContext` object
 * @param lp `LineProcessor` 
 * @author Ben Wellner
 */ 
class FileBasedEvaluator[IType : ClassTag, U : ClassTag, W <: Weights[W], R : ClassTag, OType : ClassTag, C <: Confusion[C] : ClassTag](
    _fe: FeatureExtractor[IType,U],
    _pr: EvalPredictor[U, W, R, C],
    _oc: OutputConstructor[IType, R, OType],
    val sc: SparkContext,
    val lp: LineProcessor[IType, OType]) extends EvalDecoder[IType, U, W, R, C](_fe, _pr) {
  
  def evaluateFile(ifile: String, ofile: String, wBc: Broadcast[W], numPartitions: Int) : Unit = {
    val _lp = lp
    val _sc = sc
    val _evDecoder = eval _
    val lines      = _sc.textFile(ifile, numPartitions)
    
    val lineToInput = _lp.lineToInput _
    val inputs      = lines map lineToInput
    val confMat     = _evDecoder(inputs, wBc)
    val dim         = confMat.dim
    val labels      = (0 until dim) map _oc.intToResponseString    
    confMat.prettyPrint(labels.toArray, new java.io.File(ofile))
  }
}

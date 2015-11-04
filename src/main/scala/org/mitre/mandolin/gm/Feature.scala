package org.mitre.mandolin.gm
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

abstract class FeatureCore

/**
 * Feature class; includes an integer indexing the feature. Assumes
 * feature is unit valued (common for features in certain domains)
 * @param fid integer id for the feature in parameter space
 */
class Feature(val fid: Int) {  
  def value : Double = 1.0
  override def hashCode : Int = fid
  override def equals(th: Any) = th match {case th: Feature => th.fid == this.fid case _ => false }
}

/**
 * Represents features with non-unit value
 * @param fid feature id
 * @param value feature value
 * @author wellner
 */
class NonUnitFeature(_fid: Int, override val value : Double) extends Feature(_fid)

class OneVariableFeature(val state: Int, fid: Int) extends Feature(fid)

/**
 * @param state - integer id for the categorical value associated with this feature
 * @param fid - feature id
 * @param value - value of the feature 
 */ 
class OneVariableNonUnitFeature(state: Int, fid: Int, override val value: Double) extends OneVariableFeature(state, fid) 


/**
 * Represents features within factors of two variables
 */
class TwoVariableFeature(val state1: Int, val state2: Int, fid: Int) extends Feature(fid)

/**
 * @param state1 - integer id for the categorical value associated with "left" variable of factor this feature parameterizes
 * @param state2 - integer id for the categorical value associated with "right" variable of factor this feature parameterizes
 * @param fid - feature id
 * @param vl - value of the feature 
 */ 
class TwoVariableNonUnitFeature(state1: Int, state2: Int, fid: Int, val vl: Double) extends TwoVariableFeature(state1, state2, fid) {
  override val value = vl
}

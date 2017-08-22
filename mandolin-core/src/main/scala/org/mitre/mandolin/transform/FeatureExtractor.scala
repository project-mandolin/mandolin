package org.mitre.mandolin.transform
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.util.{Alphabet, IdentityAlphabet}

abstract class FeatureExtractor[IType, U] extends Serializable {
  
  var noLabels = false
  def extractFeatures(input: IType) : U
  def getNumberOfFeatures : Int
  def getAlphabet : Alphabet
}

class IdentityFeatureExtractor[U] extends FeatureExtractor[U,U] {
  def extractFeatures(i: U) : U = i
  def getNumberOfFeatures = -1
  def getAlphabet = new IdentityAlphabet
}





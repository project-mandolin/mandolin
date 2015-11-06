package org.mitre.mandolin.transform
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.util.Alphabet

abstract class FeatureExtractor[IType, U] extends Serializable {
  def extractFeatures(input: IType) : U
  def getNumberOfFeatures : Int
  def getAlphabet : Alphabet
}



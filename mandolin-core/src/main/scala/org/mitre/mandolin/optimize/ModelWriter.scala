package org.mitre.mandolin.optimize
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

abstract class ModelWriter[W <: Weights[W]] {

  def writeModel(w: W) : Unit
  def writeModel(path: String, w: W) : Unit = {
    val f = new java.io.File(path)
    writeModel(f, w)
  } 
    
  def writeModel(f: java.io.File, w: W) : Unit
}

package org.mitre.mandolin.transform
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

/**
 * Handles mapping input lines (as strings) to application input representation
 * type and maps output representation type to a single string.
 * @author Ben Wellner
 */ 
abstract class LineProcessor[IType, OType] extends Serializable {
  /** Maps a single line of input file(s) to an input representation */
  def lineToInput(s: String) : IType
  
  /** Maps a model output (prediction) to a string on a single line. This may also 
   *  include auxiliary information, a full posterior distribution, etc. */
  def outputToLine(o: OType) : String    
}

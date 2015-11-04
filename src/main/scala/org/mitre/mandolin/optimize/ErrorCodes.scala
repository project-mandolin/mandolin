package org.mitre.mandolin.optimize
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

/** This object contains various error codes for the optimizer and line search routines.  
 */ 
object ErrorCodes {
  
  abstract class OptimizerStatus {
    def isError = !(this == Success || this == Stopped || this == AlreadyMinimized || this == NotStarted)
  }
  
  case object Success extends OptimizerStatus
  case object Stopped extends OptimizerStatus
  case object AlreadyMinimized extends OptimizerStatus
  case object NotStarted extends OptimizerStatus
  case object ErrUnknownerror extends OptimizerStatus
  case object ErrLogicerror extends OptimizerStatus
  case object ErrOutofmemory extends OptimizerStatus
  case object ErrCanceled extends OptimizerStatus
  case object ErrInvalidN extends OptimizerStatus
  case object ErrInvalidNSse extends OptimizerStatus
  case object ErrInvalidXSse extends OptimizerStatus
  case object ErrInvalidEpsilon extends OptimizerStatus
  case object ErrInvalidTestperiod extends OptimizerStatus
  case object ErrInvalidDelta extends OptimizerStatus
  case object ErrInvalidLinesearch extends OptimizerStatus
  case object ErrInvalidMinstep extends OptimizerStatus
  case object ErrInvalidMaxstep extends OptimizerStatus
  case object ErrInvalidFtol extends OptimizerStatus
  case object ErrInvalidWolfe extends OptimizerStatus
  case object ErrInvalidGtol extends OptimizerStatus
  case object ErrInvalidXtol extends OptimizerStatus
  case object ErrInvalidMaxlinesearch extends OptimizerStatus
  case object ErrInvalidOrthantwise extends OptimizerStatus
  case object ErrInvalidOrthantwiseStart extends OptimizerStatus
  case object ErrInvalidOrthantwiseEnd extends OptimizerStatus
  case object ErrOutofinterval extends OptimizerStatus
  case object ErrIncorrectTminmax extends OptimizerStatus
  case object ErrRoundingError extends OptimizerStatus
  case object ErrMinimumstep extends OptimizerStatus
  case object ErrMaximumstep extends OptimizerStatus
  case object ErrMaximumlinesearch extends OptimizerStatus
  case object MaximumIteration extends OptimizerStatus
  case object SubMaxIteration extends OptimizerStatus
  case object ErrWidthtoosmall extends OptimizerStatus
  case object ErrInvalidparameters extends OptimizerStatus
  case object ErrIncreasegradient extends OptimizerStatus
  
  abstract class LineSearchAlg
  case object LineSearchBacktracking extends LineSearchAlg
  case object LineSearchMoreThuente extends LineSearchAlg
  case object LineSearchBacktrackingArmijo extends LineSearchAlg
  case object LineSearchBacktrackingWolfe extends LineSearchAlg


}

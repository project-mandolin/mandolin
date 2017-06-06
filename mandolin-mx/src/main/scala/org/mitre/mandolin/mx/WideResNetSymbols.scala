package org.mitre.mandolin.mx

import ml.dmlc.mxnet._

object WideResNetSymbols {
  
  
  def residualUnit(data: Symbol, numFilter: Int, stride: String, dimMatch: Boolean, bottleNeck: Boolean, workSpace: Int) : Symbol = {
    if (bottleNeck) {
      val bn1 = Symbol.BatchNorm()()(Map("data" -> data, "fix_gamma" -> "False", "eps" -> "2e-5"))
      val act1 = Symbol.Activation()()(Map("data" -> bn1, "act_type" -> "relu"))
      val conv1 = Symbol.Convolution()()(Map("data" -> act1, "num_filter" -> (numFilter * 0.25).toInt.toString, "kernel" -> "(1,1)",
          "stride" -> "(1,1)", "pad" -> "(0,0)", "no_bias" -> "True", "workspace" -> workSpace.toString))
      val bn2 = Symbol.BatchNorm()()(Map("data" -> conv1, "fix_gamma" -> "False", "eps" -> "2e-5"))
      val act2 = Symbol.Activation()()(Map("data" -> bn2, "act_type" -> "relu"))
      val conv2 = Symbol.Convolution()()(Map("data" -> act2, "num_filter" -> (numFilter * 0.25).toInt.toString, "kernel" -> "(3,3)",
          "stride" -> stride, "pad" -> "(1,1)", "no_bias" -> "True", "workspace" -> workSpace.toString))
      val bn3 = Symbol.BatchNorm()()(Map("data" -> conv2, "fix_gamma" -> "False", "eps" -> "2e-5"))
      val act3 = Symbol.Activation()()(Map("data" -> bn3, "act_type" -> "relu"))
      val conv3 = Symbol.Convolution()()(Map("data" -> act3, "num_filter" -> numFilter.toString, "kernel" -> "(1,1)",
          "stride" -> "(1,1)", "pad" -> "(0,0)", "no_bias" -> "True", "workspace" -> workSpace.toString))
      val shortCut = if (dimMatch) data else Symbol.Convolution()()(Map("data" -> act1, "num_filter" -> numFilter, "kernel" -> "(1,1)",
          "stride" -> stride, "no_bias" -> "True", "workspace" -> workSpace.toString))
      conv3 + shortCut
    } else {
      val bn1 = Symbol.BatchNorm()()(Map("data" -> data, "fix_gamma" -> "False", "eps" -> "2e-5"))
      val act1 = Symbol.Activation()()(Map("data" -> bn1, "act_type" -> "relu"))
      val conv1 = Symbol.Convolution()()(Map("data" -> act1, "num_filter" -> numFilter.toInt.toString, "kernel" -> "(3,3)",
          "stride" -> stride, "pad" -> "(1,1)", "no_bias" -> "True", "workspace" -> workSpace.toString))
      val bn2 = Symbol.BatchNorm()()(Map("data" -> conv1, "fix_gamma" -> "False", "eps" -> "2e-5"))
      val act2 = Symbol.Activation()()(Map("data" -> bn2, "act_type" -> "relu"))
      val conv2 = Symbol.Convolution()()(Map("data" -> act2, "num_filter" -> numFilter.toInt.toString, "kernel" -> "(3,3)",
          "stride" -> "(1,1)", "pad" -> "(1,1)", "no_bias" -> "True", "workspace" -> workSpace.toString))
      val shortCut = if (dimMatch) data else Symbol.Convolution()()(Map("data" -> act1, "num_filter" -> numFilter, "kernel" -> "(1,1)",
          "stride" -> stride, "no_bias" -> "True", "workspace" -> workSpace.toString))
      conv2 + shortCut
    }
  }
  
  def resnetV2(in: Symbol, units: List[Int], filterList: List[Int], bottleNeck: Boolean, workSpace: Int) : Symbol = {
    val numStages = units.length
    // this will get done via spec file
    /*
    val data = Symbol.Variable("data")
    val bn1  = Symbol.BatchNorm()(Map("data" -> data, "eps" -> "2e-5", "fix_gamma" -> "True" ))
    val conv1 = Symbol.Convolution()(Map("data" -> bn1, "num_filter"))
    * 
    */
    // iterate over stages here, some hard-coding
    var curSym = in
    for (i <- 0 until units.length) {
      val stride = if (i < 1) "(1,1)" else "(2,2)"
      curSym = residualUnit(curSym, filterList(i+1), stride, false, bottleNeck, workSpace)
      for (j <- 0 until (units(i) - 1)) {
        curSym = residualUnit(curSym, filterList(i+1), "(1,1)", true, bottleNeck, workSpace)
      }
    }
    /*
    val bn = Symbol.BatchNorm()(Map("data" -> curSym, "fix_gamma" -> "False", "eps" -> "2e-5"))
    val relu = Symbol.Activation()(Map("data" -> bn, "act_type" -> "relu"))
    val pool = Symbol.Pooling()(Map("data" -> relu, "global_pool" -> "True", "kernel" -> "(7,7)", "pool_type" -> "avg"))
    val flat = Symbol.Flatten()(Map("data" -> pool))
    */
    curSym
  }

}
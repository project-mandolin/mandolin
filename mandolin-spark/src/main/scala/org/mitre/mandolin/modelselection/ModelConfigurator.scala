package org.mitre.mandolin.modelselection

import org.apache.spark.ml.param.{ParamMap, ParamValidators, Param, Params}
import org.apache.spark.ml.tuning.ParamGridBuilder

/**
  * Created by jkraunelis on 10/4/16.
  */

trait HasSGDMethod extends Params {
  final val sgdMethod = new Param[String](this, "SGDMethod", "todo", ParamValidators.inArray(Array("adagrad", "sgd")))

  final def getSgdMethod: String = $(sgdMethod)
}

trait HasInitialLearningRate extends Params {
  final val initialLearningRate = new Param[Float](this, "InitialLearningRate", "todo", ParamValidators.gtEq(0))

  final def getInitialLearningRate: Float = $(initialLearningRate)
}

trait HasMParam extends Params {
  final val mParam = new Param[Int](this, "m", "todo", ParamValidators.gtEq(0))

  final def getmParam: Int = $(mParam)
}

class MSTest extends HasSGDMethod with HasInitialLearningRate with HasMParam {
  override def copy(extra: ParamMap): Params = defaultCopy(extra)

  override val uid: String = "xyz"
}

object ModelConfigurator {
  def main(args: Array[String]): Unit = {
    val msTest = new MSTest
    val paramGrid = new LazyParamGridBuilder()
    paramGrid.addGrid(msTest.sgdMethod, Array("adagrad", "sgd"))
    paramGrid.addGrid(msTest.initialLearningRate, (0.0 to 1.0 by 0.1).toArray.map(_.toFloat))
    paramGrid.addGrid(msTest.mParam, Array(6, 66, 666))

    println(paramGrid.toIterator.map{ pm => pm } )
  }

  /* val lineSearch: LineSearchAlg = LineSearchBacktracking,
   var m: Int = 6,
   var epsilon: Double = 1E-5,
   var past: Int = 3,
   var delta: Double = 1E-5,
   var maxIters: Int = 0, // value of '0' indicates iterate until convergence
   var maxLineSearch: Int = 40,
   var minStep: Double = 1E-20,
   var maxStep: Double = 1E20,
   var ftol: Double = 1E-4,
   var wolfe: Double = 0.9,
   var gtol: Double = 0.9,
   var xtol: Double = 1E-16,
   var verbose: Boolean = false,
   var veryVerbose: Boolean = false,
   var intermediateModelWriter: Option[(Array[Double], Int) => Unit] = None)*/

  //optimizer {
  //method                = adagrad
  //initial-learning-rate = 0.1

}

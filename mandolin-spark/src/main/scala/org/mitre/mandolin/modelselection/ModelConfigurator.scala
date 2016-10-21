package org.mitre.mandolin.modelselection

import org.apache.spark.ml.param.{ParamMap, ParamValidators, Param, Params}
import org.mitre.mandolin.optimize.ErrorCodes._

/**
  * Created by jkraunelis on 10/4/16.
  */

/*trait HasLineSearchAlgorithm extends Params {
  val lineSearchAlgs = Array(LineSearchBacktracking, LineSearchMoreThuente, LineSearchBacktrackingArmijo, LineSearchBacktrackingWolfe)

  final val lineSearchAlgorithm = new Param[LineSearchAlg](this, "lineSearchAlgorithm", "todo", ParamValidators.inArray(lineSearchAlgs))

  final def getLineSearchAlgorithm: LineSearchAlg = $(lineSearchAlgorithm)
}*/


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
    val mcb = new ModelConfigurationBuilder()
    mcb.addGrid(msTest.sgdMethod, Array("adagrad", "sgd"))
    mcb.addGrid(msTest.initialLearningRate, (0.0 to 1.0 by 0.1).toArray.map(_.toFloat))
    mcb.addGrid(msTest.mParam, Array(6, 67, 678))

    mcb.toIterator.map{ pm => pm }.foreach(println)
  }

  /*

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



}

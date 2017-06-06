package org.mitre.mandolin.xg

import org.mitre.mandolin.glp.{GLPFactor, StdGLPFactor, SparseGLPFactor}
import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost}

class XGBoostEvaluator(settings: XGModelSettings) {
  
  val paramMap = new scala.collection.mutable.HashMap[String, Any]
  paramMap.put("gamma", settings.gamma)
  paramMap.put("max_depth", settings.maxDepth)
  paramMap.put("objective", settings.objective)
  paramMap.put("scale_pos_weight", settings.scalePosWeight)

  def mapGLPFactorToLabeledPoint(gf: GLPFactor) : LabeledPoint = {
    gf match {
      case x: SparseGLPFactor =>
        val spv = x.getInput
        val out = x.getOutput
        val outVal = out.argmax.toFloat
        LabeledPoint.fromSparseVector(outVal, spv.indArray, spv.valArray)
      case x: StdGLPFactor =>
        val dv = x.getInput
        val out = x.getOutput
        val outVal = out.argmax.toFloat
        LabeledPoint.fromDenseVector(outVal, dv.asArray)
    }
  }
  
  def evaluateTrainingSet(train: Iterator[GLPFactor], test: Option[Iterator[GLPFactor]]) = {
    val trIter = train map mapGLPFactorToLabeledPoint
    val tstIter = test map {iter => iter map mapGLPFactorToLabeledPoint }
    val trainDm = new DMatrix(trIter)
    tstIter match {
      case Some(tst) => 
        XGBoost.train(trainDm, paramMap.toMap, settings.rounds, Map("auc" -> new DMatrix(tst)))
      case None => XGBoost.train(trainDm, paramMap.toMap, settings.rounds)
    }
  }
  
  
}
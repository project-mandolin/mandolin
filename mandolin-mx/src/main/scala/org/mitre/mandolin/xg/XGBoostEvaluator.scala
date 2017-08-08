package org.mitre.mandolin.xg

import org.mitre.mandolin.mlp.{MMLPFactor, StdMMLPFactor, SparseMMLPFactor}
import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost, Booster}

class XGBoostEvaluator(settings: XGModelSettings, numLabels: Int) {
  
  val paramMap = new scala.collection.mutable.HashMap[String, Any]
  paramMap.put("gamma", settings.gamma)
  paramMap.put("max_depth", settings.maxDepth)
  paramMap.put("objective", settings.objective)
  paramMap.put("scale_pos_weight", settings.scalePosWeight)
  paramMap.put("silent", settings.silent)
  paramMap.put("eval_metric", settings.evalMethod)
  paramMap.put("nthread", settings.numThreads)
  if (numLabels > 2) paramMap.put("num_class", numLabels)  

  def mapMMLPFactorToLabeledPoint(gf: MMLPFactor) : LabeledPoint = {
    gf match {
      case x: SparseMMLPFactor =>
        val spv = x.getInput
        val out = x.getOutput
        val outVal = out.argmax.toFloat
        LabeledPoint.fromSparseVector(outVal, spv.indArray, spv.valArray)
      case x: StdMMLPFactor =>
        val dv = x.getInput
        val out = x.getOutput
        val outVal = out.argmax.toFloat
        LabeledPoint.fromDenseVector(outVal, dv.asArray)
    }
  }

  def gatherTestAUC(s: String) = {
    s.split('\t').toList match {
      case rnd :: tst :: _ => tst.split(':')(1).toFloat
      case _ => throw new RuntimeException("Unable to parse cross validation metric: " + s)
    }
  }

  def evaluateTrainingSet(train: Iterator[MMLPFactor], test: Option[Iterator[MMLPFactor]]) : (Float, Option[Booster]) = {
    val trIter = train map mapMMLPFactorToLabeledPoint
    val tstIter = test map {iter => iter map mapMMLPFactorToLabeledPoint }
    val trainDm = new DMatrix(trIter)
    tstIter match {
      case Some(tst) =>
        val metrics = Array(Array.fill(settings.rounds)(0.0f))
        val b = XGBoost.train(trainDm, paramMap.toMap, settings.rounds, Map("auc" -> new DMatrix(tst)), metrics, null, null)
        // val xv = XGBoost.crossValidation(trainDm, paramMap.toMap, settings.rounds, 5, Array("auc"), null, null)        
        // val finalTestMetric = gatherTestAUC(xv.last)
        (1.0f - metrics(0)(settings.rounds - 1), Some(b))
      case None => 
        val xv = XGBoost.crossValidation(trainDm, paramMap.toMap, settings.rounds, 5, Array("auc"), null, null)
        val finalTestMetric = gatherTestAUC(xv.last)
        (1.0f - finalTestMetric, None)
    }
  }
  
  
}
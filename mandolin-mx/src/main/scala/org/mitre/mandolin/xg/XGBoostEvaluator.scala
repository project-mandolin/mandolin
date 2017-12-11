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
  paramMap.put("eta", settings.eta)
  paramMap.put("booster", settings.booster)
  
  val nl = if (settings.regression) 1 else numLabels
  if (settings.regression) paramMap.put("num_class", 1)
  if (nl > 2) paramMap.put("num_class", numLabels)  

  def mapMMLPFactorToLabeledPoint(gf: MMLPFactor) : LabeledPoint = {
    gf match {
      case x: SparseMMLPFactor =>
        val spv = x.getInput
        val out = x.getOutput        
        val outVal = if (settings.regression) out(0) else out.argmax.toFloat
        LabeledPoint.fromSparseVector(outVal, spv.indArray, spv.valArray)
      case x: StdMMLPFactor =>
        val dv = x.getInput
        val out = x.getOutput
        val outVal = if (settings.regression) out(0) else out.argmax.toFloat
        LabeledPoint.fromDenseVector(outVal, dv.asArray)
    }
  }

  def gatherTestAUC(s: String) = {
    s.split('\t').toList match {
      case rnd :: tst :: _ => tst.split(':')(1).toFloat
      case _ => throw new RuntimeException("Unable to parse cross validation metric: " + s)
    }
  }
  
  def getPredictionsAndEval(booster: Booster, data: Iterator[MMLPFactor], evMethod: String = "auc") : (Array[Array[Float]], Float) = {
    val dataDm = new DMatrix(data map mapMMLPFactorToLabeledPoint)    
    val res = booster.predict(dataDm,false,0)
    val evalInfo = booster.evalSet(Array(dataDm), Array(evMethod), 1)
    (res, 1.0f - gatherTestAUC(evalInfo))
  }

  def evaluateTrainingSet(train: Iterator[MMLPFactor], test: Option[Iterator[MMLPFactor]]) : (Float, Option[Booster]) = {
    val trIter = train map mapMMLPFactorToLabeledPoint
    val tstIter = test map {iter => iter map mapMMLPFactorToLabeledPoint }
    val trainDm = new DMatrix(trIter)
    tstIter match {
      case Some(tst) =>
        val metrics = Array(Array.fill(settings.rounds)(0.0f))
        val b = XGBoost.train(trainDm, paramMap.toMap, settings.rounds, Map("auc" -> new DMatrix(tst)), metrics, null, null)
        (1.0f - metrics(0)(settings.rounds - 1), Some(b))
      case None =>
        if (settings.appMode equals "train-test") {
          val metricEval = settings.evalMethod
          val xv = XGBoost.crossValidation(trainDm, paramMap.toMap, settings.rounds, 5, Array(metricEval), null, null)
          val finalTestMetric = gatherTestAUC(xv.last)
          (1.0f - finalTestMetric, None)
        } else {
          val metrics = Array(Array.fill(settings.rounds)(0.0f))
          val b = XGBoost.train(trainDm, paramMap.toMap, settings.rounds, Map("auc" -> trainDm), metrics, null, null)
          (1.0f - metrics(0)(settings.rounds - 1), Some(b))
        }
    }
  }
  
  
}
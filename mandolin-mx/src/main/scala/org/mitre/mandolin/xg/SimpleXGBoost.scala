package org.mitre.mandolin.xg

import ml.dmlc.xgboost4j.scala.{ DMatrix, XGBoost }
import scala.collection.mutable

class SimpleXGBoost {

}

object SimpleXGBoost {
    def main(args: Array[String]) {
     // read trainining data, available at xgboost/demo/data
     val inFile = args(0)
     val iters  = args(1).toInt
     val maxDepth = if (args.length > 2) args(2).toInt else 3
     val scalePos = if (args.length > 3) args(3).toDouble else 0.5
     val etaVal = if (args.length > 4) args(4).toDouble else 0.3
     val alpha = if (args.length > 5) args(5).toDouble else 0.0
     val lambda = if (args.length > 6) args(6).toDouble else 1.0
     val trainMat = new DMatrix(inFile)

    // define parameters
    /*
     val paramMap = List(
      "eta" -> 0.1,
      "max_depth" -> 2,
      "objective" -> "binary:logistic").toMap
    */

    val params = new mutable.HashMap[String, Any]

    params.put("eta", etaVal)
    params.put("alpha", alpha)
    params.put("lambda", lambda)
    params.put("max_depth", maxDepth)
    params.put("silent", 1)
    params.put("nthread", 6)
    // params.put("objective", "binary:logistic")

    // params.put("objective", "multi:softmax")
    // params.put("num_class", 4)
    //params.put("objective", "rank:pairwise")
    params.put("objective", "binary:logistic")
    params.put("scale_pos_weight", scalePos)

    // params.put("gamma", 10.0)
    // params.put("eval_metric", "merror")
    params.put("eval_metric", "auc")


    val round: Int = iters
    val nfold: Int = 5
    // set additional eval_metrics
    val metrics: Array[String] = null

    val evalHist: Array[String] = XGBoost.crossValidation(trainMat, params.toMap, round, nfold, metrics, null, null)
  }

}
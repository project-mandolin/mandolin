package org.mitre.mandolin.xg

import org.mitre.mandolin.mlp.{MMLPFactor, StdMMLPFactor, SparseMMLPFactor}
import org.mitre.mandolin.util.{LocalIOAssistant, AbstractPrintWriter, Alphabet, IdentityAlphabet}
import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost, Booster}

trait DataConversions {
  def mapMMLPFactorToLabeledPoint(regression: Boolean)(gf: MMLPFactor) : LabeledPoint = {
    gf match {
      case x: SparseMMLPFactor =>
        val spv = x.getInput
        val out = x.getOutput        
        val outVal = if (regression) out(0) else out.argmax.toFloat
        LabeledPoint.fromSparseVector(outVal, spv.indArray, spv.valArray)
      case x: StdMMLPFactor =>
        val dv = x.getInput
        val out = x.getOutput
        val outVal = if (regression) out(0) else out.argmax.toFloat
        LabeledPoint.fromDenseVector(outVal, dv.asArray)
    }
  }
}

class XGBoostEvaluator(settings: XGModelSettings, numLabels: Int) extends DataConversions {
  
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

  

  def gatherTestAUC(s: String) = {
    s.split('\t').toList match {
      case rnd :: tst :: _ => tst.split(':')(1).toFloat
      case _ => throw new RuntimeException("Unable to parse cross validation metric: " + s)
    }
  }
  
  def getPredictionsAndEval(booster: Booster, data: Iterator[MMLPFactor], evMethod: String = "auc") : (Array[Array[Float]], Float) = {
    val dataDm = new DMatrix(data map mapMMLPFactorToLabeledPoint(settings.regression))    
    val res = booster.predict(dataDm,false,0)
    val evalInfo = booster.evalSet(Array(dataDm), Array(evMethod), 1)
    (res, 1.0f - gatherTestAUC(evalInfo))
  }

  def evaluateTrainingSet(trainVec: Vector[MMLPFactor], test: Option[Iterator[MMLPFactor]]) : (Float, Option[Booster]) = {
    val train = trainVec.toIterator
    val trIter = train map mapMMLPFactorToLabeledPoint(settings.regression)
    val tstIter = test map {iter => iter map mapMMLPFactorToLabeledPoint(settings.regression) }
    val trainDm = new DMatrix(trIter)
    tstIter match {
      case Some(tst) =>
        val metrics = Array(Array.fill(settings.rounds)(0.0f))
        val b = XGBoost.train(trainDm, paramMap.toMap, settings.rounds, Map("auc" -> new DMatrix(tst)), metrics, null, null)
        (1.0f - metrics(0)(settings.rounds - 1), Some(b))
      case None =>
        if (settings.appMode equals "train-test") {
          val metricEval = settings.evalMethod
          if (settings.outputFile.isDefined) {
            getCrossValidationPredictions(trainVec, paramMap, settings.rounds, 5, settings.outputFile.get)
            (0.0f, None)
          } else {
            val xv = XGBoost.crossValidation(trainDm, paramMap.toMap, settings.rounds, 5, Array(metricEval), null, null)
            val finalTestMetric = gatherTestAUC(xv.last)
            (1.0f - finalTestMetric, None)
          }
        } else {
          val metrics = Array(Array.fill(settings.rounds)(0.0f))
          val b = XGBoost.train(trainDm, paramMap.toMap, settings.rounds, Map("auc" -> trainDm), metrics, null, null)
          (1.0f - metrics(0)(settings.rounds - 1), Some(b))
        }
    }
  }
  
  def getCrossValidationPredictions(allDataFactors: Vector[MMLPFactor], paramMap: collection.mutable.HashMap[String, Any], rounds: Int, folds: Int, file: String) = {
    val io = new LocalIOAssistant
    val allData = allDataFactors map mapMMLPFactorToLabeledPoint(settings.regression)
    val totalSize = allData.length
    val testFoldSize = totalSize / folds
    val nums = util.Random.shuffle(for (i <- 0 until totalSize) yield i)
    val os = io.getPrintWriterFor(file, false)
    val trTstSplits = for (i <- 1 to folds) yield {
      var trId = 0
      var tstId = 0
      val tstItems = Array.fill[Int](testFoldSize)(0)
      val trItems = Array.fill[Int](totalSize - testFoldSize)(0)
      for (j <- 0 until totalSize) {
        if ((j >= (i * testFoldSize)) && (j < (i * testFoldSize + testFoldSize)) && (tstId < testFoldSize)) {
          tstItems(tstId) = nums(j)
          tstId += 1
        } else {
          if (trId < (totalSize - testFoldSize)) {
            trItems(trId) = nums(j)
            trId += 1
          } else {
            tstItems(tstId) = nums(j)
            tstId += 1
          }
        }
      }
      val dTrain = Vector.tabulate(trItems.length){i => allData(trItems(i))}
      val dTest  = Vector.tabulate(tstItems.length){i => allData(tstItems(i))}
      println("dTrain size = " + dTrain.length)
      println("dTest size = " + dTest.length)
      (dTrain, dTest)
    }
    trTstSplits foreach { case (tr,tst) =>
      val dTrain = new DMatrix(tr.toIterator)
      val dTest  = new DMatrix(tst.toIterator)
      println("About to train with training set (size = " + dTrain.rowNum + ")")     
      val metrics = Array(Array.fill(rounds)(0.0f))
      val booster = XGBoost.train(dTrain, paramMap.toMap, rounds, Map("auc" -> dTrain), metrics, null, null)
      println("About to eval with test set (size = " + dTest.rowNum + ")")
      val res = booster.predict(dTest, false, 0)
      
      var k = 0; while (k < res.length) {
        val posterior = res(k)
        posterior foreach {p =>
          os.write(p.toString)
          os.write(',')
        }
        os.write(tst(k).label.toString)
        os.println
        k += 1
      }
    }
    os.close()
  }
  
  
  
  
}
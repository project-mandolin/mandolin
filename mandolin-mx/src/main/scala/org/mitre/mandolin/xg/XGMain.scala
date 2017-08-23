package org.mitre.mandolin.xg

import org.mitre.mandolin.mlp.{MMLPFactor, MMLPTrainerBuilder}
import org.mitre.mandolin.util.{LocalIOAssistant, AbstractPrintWriter, Alphabet}
import org.mitre.mandolin.transform.FeatureExtractor
import ml.dmlc.xgboost4j.scala.{XGBoost, DMatrix}

object PrintUtils {
  def writeOutputs(os: AbstractPrintWriter, outputs: Iterator[(String, MMLPFactor)], laOpt: Option[Alphabet], noLabels: Boolean = false): Unit = {
    laOpt foreach { la =>
      val invLa = la.getInverseMapping
      if (la.getSize < 3) {
        os.print("ID,response")
        if (!noLabels) os.print(",value") // if we have labels/values (regression), print them
        os.println
        outputs foreach {case (s, factor) => 
          os.print(s)
          if (!noLabels) {
            os.print(','); os.print(invLa(factor.getOutput(0).toInt).toString)
          }
          os.println}
      } else {
      val labelHeader = la.getMapping.toSeq.sortWith((a, b) => a._2 < b._2).map(_._1) // get label strings sorted by their index
      os.print("ID")
      for (i <- 0 until labelHeader.length) {
        os.print(',')
        os.print(labelHeader(i))
      }
      if (!noLabels) {
        os.print(',')
        os.print("Label")
      }
      os.println
      outputs foreach { case (s, factor) => 
        os.print(s)
        if (!noLabels) { os.print(','); os.print(invLa(factor.getOneHot.toInt).toString) } // pass through ground truth if we had it 
        os.println }
      }
    }
  }
}

object XGMain extends org.mitre.mandolin.config.LogInit with org.mitre.mandolin.app.AppMain {


  def getVecIOs(appSettings: XGModelSettings, modelSpec: Option[XGBoostModelSpec], noLabels: Boolean = false): (Vector[MMLPFactor], Option[Vector[MMLPFactor]], FeatureExtractor[String, MMLPFactor], Alphabet) = {
    val io = new LocalIOAssistant
    val (featureExtractor, labelAlphabet) = modelSpec match {
      case None => 
        val components = MMLPTrainerBuilder.getComponentsViaSettings(appSettings, io)
        (components.featureExtractor, components.labelAlphabet)
      case Some(sp) => (sp.fe, sp.la)
    }
    featureExtractor.noLabels_=(noLabels)
    if (modelSpec.isDefined) {
      val tstFile = appSettings.testFile
      val tstVecs = tstFile map { tf => (io.readLines(tf) map { l => featureExtractor.extractFeatures(l) } toVector) }
      (Vector(), tstVecs, featureExtractor, labelAlphabet)
    } else {
      val trFile = appSettings.trainFile.get    
      val trVecs = (io.readLines(trFile) map { l => featureExtractor.extractFeatures(l) } toVector)
      val tstFile = appSettings.testFile
      val tstVecs = tstFile map { tf => (io.readLines(tf) map { l => featureExtractor.extractFeatures(l) } toVector) }      
      (trVecs, tstVecs, featureExtractor, labelAlphabet)      
    }
    
  }

  def main(args: Array[String]): Unit = {
    val xgSettings = new XGModelSettings(args)
    val mode = xgSettings.appMode    
    mode match {
      case "train" =>
        val (trVecs, tstVecs, fe, labelAlphabet) = getVecIOs(xgSettings, None)
        val evaluator = new XGBoostEvaluator(xgSettings, labelAlphabet.getSize)
        val (finalMetric, booster) = evaluator.evaluateTrainingSet(trVecs.toIterator, tstVecs map { _.toIterator})
        val io = new LocalIOAssistant
        booster foreach { b =>          
          xgSettings.modelFile match {            
          case Some(m) =>
            val writer = new StandaloneXGBoostModelWriter
            writer.writeModel(io, m, b, labelAlphabet, fe)
          case None => throw new RuntimeException("Destination model file path expected")
          }
        }
      case "train-test" =>
        val (trVecs, tstVecs, fe, labelAlphabet) = getVecIOs(xgSettings, None)
        val evaluator = new XGBoostEvaluator(xgSettings, labelAlphabet.getSize)
        val (finalMetric, _) = evaluator.evaluateTrainingSet(trVecs.toIterator, tstVecs map { _.toIterator})
        println("Training complete.")
        println("Final evaluation metric result: " + finalMetric)
      case "predict-eval" =>
        val mfile = xgSettings.modelFile.get
        val reader = new StandaloneXGBoostModelReader
        val io = new LocalIOAssistant
        val spec = reader.readModel(mfile, io)
        val (_, tstVecs, _, labelAlphabet) = getVecIOs(xgSettings, Some(spec))
        val evaluator = new XGBoostEvaluator(xgSettings, spec.la.getSize)
        val model = XGBoost.loadModel(new java.io.ByteArrayInputStream(spec.booster))
        val (results, metric) = evaluator.getPredictionsAndEval(model, tstVecs.get.toIterator)    
        val resVecAndFactor = results.toVector map {ar =>
          val buf = new StringBuilder
          var i = 0; while (i < ar.length) {
            if (i > 0) buf append ","
            buf append ar(i).toString
            i += 1
          }
          buf.toString
        } zip tstVecs.get
        val finalOutputs = resVecAndFactor map {case (s,f) =>
          val sb = new StringBuilder
          sb append f.getUniqueKey.getOrElse("-1")
          sb append ","
          sb append s
          (sb.toString, f)
          }        
        val os = io.getPrintWriterFor(xgSettings.outputFile.get, false)
        PrintUtils.writeOutputs(os, finalOutputs.toIterator, Some(labelAlphabet), false)
        os.close()
      case "predict" =>
        val mfile = xgSettings.modelFile.get
        val reader = new StandaloneXGBoostModelReader
        val io = new LocalIOAssistant
        val spec = reader.readModel(mfile, io)
        val (_, tstVecs, _, labelAlphabet) = getVecIOs(xgSettings, Some(spec), true)
        val evaluator = new XGBoostEvaluator(xgSettings, spec.la.getSize)
        val model = XGBoost.loadModel(new java.io.ByteArrayInputStream(spec.booster))
        val (results, metric) = evaluator.getPredictionsAndEval(model, tstVecs.get.toIterator)    
        val resVecAndFactor = results.toVector map {ar =>
          val buf = new StringBuilder
          var i = 0; while (i < ar.length) {
            if (i > 0) buf append ","
            buf append ar(i).toString
            i += 1
          }
          buf.toString
        } zip tstVecs.get
        val finalOutputs = resVecAndFactor map {case (s,f) =>
          val sb = new StringBuilder
          sb append f.getUniqueKey.getOrElse("-1")
          sb append ","
          sb append s
          (sb.toString, f)
          }        
        val os = io.getPrintWriterFor(xgSettings.outputFile.get, false)
        PrintUtils.writeOutputs(os, finalOutputs.toIterator, Some(labelAlphabet), true)
        os.close()
      case a => throw new RuntimeException("Unknown mandolin.mode = " + a)
    }
    
    
    
    
    
  }

}
package org.mitre.mandolin.xg

import org.mitre.mandolin.mlp.{MMLPFactor, MMLPTrainerBuilder}
import org.mitre.mandolin.util.{LocalIOAssistant, AbstractPrintWriter, Alphabet}
import ml.dmlc.xgboost4j.scala.{XGBoost, DMatrix}

object PrintUtils {
  def writeOutputs(os: AbstractPrintWriter, outputs: Iterator[(String, MMLPFactor)], laOpt: Option[Alphabet], noLabels: Boolean = false): Unit = {
    laOpt foreach { la =>
      val invLa = la.getInverseMapping
      if (la.getSize < 2) {
        os.print("ID,response,value\n")
        outputs foreach {case (s, factor) => os.print(s); os.print(','); os.print(invLa(factor.getOutput(0).toInt).toString); os.println}
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


  def getVecIOs(appSettings: XGModelSettings): (Vector[MMLPFactor], Option[Vector[MMLPFactor]], Int, Alphabet) = {
    val io = new LocalIOAssistant
    val components = MMLPTrainerBuilder.getComponentsViaSettings(appSettings, io)
    val featureExtractor = components.featureExtractor
    val trFile = appSettings.trainFile.get
    val tstFile = appSettings.testFile
    val trVecs = (io.readLines(trFile) map { l => featureExtractor.extractFeatures(l) } toVector)
    val tstVecs = tstFile map { tf => (io.readLines(tf) map { l => featureExtractor.extractFeatures(l) } toVector) }
    (trVecs, tstVecs, featureExtractor.getNumberOfFeatures, components.labelAlphabet)
  }

  def main(args: Array[String]): Unit = {
    val xgSettings = new XGModelSettings(args)
    val (trVecs, tstVecs, _, labelAlphabet) = getVecIOs(xgSettings)
    val mode = xgSettings.appMode
    val evaluator = new XGBoostEvaluator(xgSettings, labelAlphabet.getSize)
    mode match {
      case "train" =>
        val (finalMetric, booster) = evaluator.evaluateTrainingSet(trVecs.toIterator, tstVecs map { _.toIterator})
        booster foreach { b =>
          xgSettings.modelFile match {
          case Some(m) => b.saveModel(xgSettings.modelFile.get)
          case None => throw new RuntimeException("Destination model file path expected")
          }
        }
      case "train-test" =>
        val (finalMetric, booster) = evaluator.evaluateTrainingSet(trVecs.toIterator, tstVecs map { _.toIterator})
        booster foreach { b =>
          xgSettings.modelFile match {
          case Some(m) => b.saveModel(xgSettings.modelFile.get)
          case None => throw new RuntimeException("Destination model file path expected")
          }
        }
        println("Training complete.")
        println("Final evaluation metric result: " + finalMetric)
      case "predict-eval" =>
        val mfile = xgSettings.modelFile.get
        val model = XGBoost.loadModel(mfile)
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
          sb append f.getId.toString
          sb append ","
          sb append s
          (sb.toString, f)
          }
        val io = new LocalIOAssistant
        val os = io.getPrintWriterFor(xgSettings.outputFile.get, false)
        PrintUtils.writeOutputs(os, finalOutputs.toIterator, Some(labelAlphabet), false)
        os.close()
      case "predict" =>
        
      case a => throw new RuntimeException("Unknown mandolin.mode = " + a)
    }
    
    
    
    
    
  }

}
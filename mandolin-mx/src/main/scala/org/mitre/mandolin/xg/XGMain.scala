package org.mitre.mandolin.xg

import org.mitre.mandolin.mlp.{MMLPFactor, MMLPTrainerBuilder}
import org.mitre.mandolin.util.{LocalIOAssistant, AbstractPrintWriter, Alphabet, IdentityAlphabet}
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
            val oo = factor.getOutput.asArray
            if (oo.length > 1) {
              val vv = factor.getOutput.argmax
              os.print(','); os.print(invLa(vv).toString)
            } else { // actual regression, just print raw value here
              os.print(','); os.print(oo(0).toString)
            }
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
  
  def getImportanceMap(dumps: Array[String], featureAlphabet: Alphabet): Map[String, Float] = {
    var mpGain = new collection.immutable.HashMap[String, Float]
    var mpCover = new collection.immutable.HashMap[String, Float]
    dumps foreach { t =>      
      t.split('\n') foreach {line =>
        val ar = line.split('[')
        if (ar.length > 1) {
          val fi1 = ar(1).split(']')
          val gain = fi1(1).split("gain=")(1).split(',')(0).toFloat
          val fid = fi1(0).split('<')(0)
          val curCover = mpCover.get(fid).getOrElse(0.0f)
          val curGain = mpGain.get(fid).getOrElse(0.0f)
          mpCover += (fid -> (curCover + 1.0f))
          mpGain += (fid -> (curGain + gain))
        }
        }
      }
    mpCover foreach {case (k,v) => mpGain += (k -> mpGain(k)/mpCover(k))}
    var finalGain = new collection.immutable.HashMap[String, Float]
    
    val featureMap = 
      featureAlphabet match {
        case fa: IdentityAlphabet => { (i: Int) => i.toString}
        case _ => featureAlphabet.getInverseMapping
    }
    
    mpGain foreach {case (k,v) =>
      val id = k.substring(1,k.length).toInt
      finalGain += (featureMap(id) -> v)
      }
    finalGain
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
          xgSettings.featureImportance foreach {fi =>
            val writer = io.getPrintWriterFor(fi, true)
            writer.print("Feature, Gain\n")
            val fscores = b.getModelDump("", true)
            val impList = getImportanceMap(fscores, fe.getAlphabet).toList.sortWith{case ((_,a), (_,b)) => a > b}
            impList foreach {case (f,g) => writer.print(f); writer.print(','); writer.print(g.toString); writer.println}
            writer.close()
            }
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
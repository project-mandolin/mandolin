package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.util.{ StdAlphabet, RandomAlphabet, Alphabet, IOAssistant }
import org.mitre.mandolin.predict.{
  DiscreteConfusion,
  EvalPredictor,
  OutputConstructor,
  Predictor
}
import org.mitre.mandolin.predict.spark.{
  EvalDecoder,
  TrainTester,
  PosteriorDecoder,
  TrainDecoder,
  Decoder,
  Trainer
}

import org.mitre.mandolin.optimize.local.LocalOnlineOptimizer
import org.mitre.mandolin.predict.local.{ LocalTrainer, LocalTrainTester, LocalTrainDecoder, LocalEvalDecoder, LocalPosteriorDecoder, LocalDecoder }

import org.mitre.mandolin.transform.{ FeatureExtractor, FeatureImportance }
import org.mitre.mandolin.gm.{ Feature, NonUnitFeature }
import org.mitre.mandolin.util.LineParser
import scala.reflect.ClassTag

/**
 * @author wellner
 */
class LocalProcessor extends Processor {        
  
  
  def processTrain(appSettings: GLPModelSettings) = {
    if (appSettings.modelFile.isEmpty) throw new RuntimeException("Model file required in training mode")
    if (appSettings.trainFile.isEmpty) throw new RuntimeException("Training file required in training mode")
    val io = new IOAssistant
    val components = getComponentsViaSettings(appSettings, io)       
    val trainFile = appSettings.trainFile
    val ev = components.evaluator
    val fe = components.featureExtractor
    val optimizer = GLPOptimizer.getLocalOptimizer(appSettings, ev)
    val lines = io.readLines(trainFile.get).toVector
    val trainer = new LocalTrainer(fe, optimizer)
    val (finalWeights,_) = trainer.trainWeights(lines)
    val modelWriter = new GLPModelWriter(None)
    modelWriter.writeModel(io, appSettings.modelFile.get, finalWeights, components.labelAlphabet, ev, fe)
    finalWeights
  }
  
  def processDecode(appSettings: GLPModelSettings) = {
    if (appSettings.modelFile.isEmpty) throw new RuntimeException("Model file required in decoding mode")
    val io = new IOAssistant
    val modelSpec = (new GLPModelReader(None)).readModel(appSettings.modelFile.get, io)
    val testLines = appSettings.testFile map { tf => io.readLines(tf).toVector }
    val evaluator = modelSpec.evaluator
    val predictor = new GLPPredictor(evaluator.glp, true)
    val oc = new GLPPosteriorOutputConstructor()
    val decoder = new LocalPosteriorDecoder(modelSpec.fe, predictor, oc)
    val os = io.getPrintWriterFor(appSettings.outputFile.get, false)
    val outputs = decoder.run(testLines.get, modelSpec.wts)
    writeOutputs(os, outputs.toIterator, Some(modelSpec.la))
    os.close()
  }
  
  def processTrainTest(appSettings: GLPModelSettings) = {
    val trainFile = appSettings.trainFile
    if (trainFile.isEmpty) throw new RuntimeException("Training file required in train-test mode")
    val io = new IOAssistant
    val components = getComponentsViaSettings(appSettings, io)
    val ev = components.evaluator
    val fe = components.featureExtractor
    val pr = components.predictor
    val optimizer = GLPOptimizer.getLocalOptimizer(appSettings, ev)
    val lines = io.readLines(trainFile.get).toVector
    val trainer = new LocalTrainer(fe, optimizer)
    val testLines = appSettings.testFile map { tf => io.readLines(tf).toVector }
    val trainTester =
          new LocalTrainTester[String, GLPFactor, GLPWeights, Int, DiscreteConfusion](trainer,
            pr, appSettings.numEpochs, appSettings.testFreq, appSettings.progressFile)
    trainTester.trainAndTest(lines, testLines.get)
  }
  
  /**
   * A convenience method that trains and evaluates N separate models and test files
   * in a parallel directory structure.
   */
  def processTrainTestDirectories(appSettings: GLPModelSettings) = {
    val dir: java.io.File = new java.io.File(appSettings.trainFile.get)
    val io = new IOAssistant
    val testDir: java.io.File = new java.io.File(appSettings.testFile.get)
    val outFile = new java.io.File(appSettings.outputFile.get)
    val trainFiles = dir.listFiles().toVector.map { _.getPath }.sorted
    val testFiles = testDir.listFiles().toVector.map { _.getPath }.sorted
    var labelsSet = false
    val os = new java.io.PrintWriter(outFile)
    (trainFiles zip testFiles) foreach {
        case (tr, tst) =>
          val tstFile = new java.io.File(tst)
          val trLines = scala.io.Source.fromFile(tr).getLines.toVector
          val components = getComponentsViaSettings(appSettings, io)
          val ev = components.evaluator
          val fe = components.featureExtractor
          val pr = components.predictor
          // XXX - this will only work properly if each train-test split has the same label set as the first fold
          val optimizer = GLPOptimizer.getLocalOptimizer(appSettings, ev)
          val lines = trLines.toVector
          val trainer = new LocalTrainer(fe, optimizer)
          val testLines = scala.io.Source.fromFile(tst).getLines().toVector
          val trainTesterDecoder =
            new LocalTrainDecoder[String, GLPFactor, GLPWeights, Int](trainer, pr, new GLPPosteriorOutputConstructor, appSettings.numEpochs)
          val outputs = trainTesterDecoder.trainAndDecode(lines, testLines)
          if (!labelsSet) {
            val labelHeader = components.labelAlphabet.getMapping.toSeq.sortWith((a, b) => a._2 < b._2).map(_._1) 
            os.print("ID")
            for (i <- 0 until labelHeader.length) {
              os.print(',')
              os.print(labelHeader(i))
            }
            os.print(',')
            os.print("Label")
            os.println
            labelsSet = true
          }
          outputs foreach { case (s, factor) => os.print(s); os.print(','); os.print(factor.getOneHot.toString); os.println }          
    }
    os.close()
  }
  
  def processTrainDecode(appSettings: GLPModelSettings) = {
    val weights = processTrain(appSettings)
    val io = new IOAssistant
    val testLines = appSettings.testFile map { tf => io.readLines(tf).toVector }
    val components = getComponentsViaSettings(appSettings, io)
    val fe = components.featureExtractor
    val predictor = components.predictor
    val oc = components.outputConstructor
    val decoder = new LocalPosteriorDecoder(fe, predictor, oc)
    val os = io.getPrintWriterFor(appSettings.outputFile.get, false)
    val outputs = decoder.run(testLines.get, weights)
    writeOutputs(os, outputs.toIterator, Some(components.labelAlphabet))
    os.close()
  }
}

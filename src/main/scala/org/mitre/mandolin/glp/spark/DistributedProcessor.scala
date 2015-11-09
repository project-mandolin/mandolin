package org.mitre.mandolin.glp.spark
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.config.AppConfig
import org.mitre.mandolin.util.{ StdAlphabet, PrescaledAlphabet, RandomAlphabet, Alphabet, IdentityAlphabet, AlphabetWithUnitScaling, IOAssistant }
import org.mitre.mandolin.optimize.spark.{
  DistributedOnlineOptimizer,
  DistributedOptimizerEstimator
}
import org.mitre.mandolin.predict.{
  DiscreteConfusion,
  EvalPredictor,
  OutputConstructor,
  Predictor
}

import org.mitre.mandolin.predict.spark.{
  EvalDecoder,
  PosteriorDecoder,
  TrainDecoder,
  TrainTester,  
  Decoder,
  Trainer
}

import org.mitre.mandolin.glp.{AbstractProcessor, GLPModelSettings, GLPOptimizer, GLPModelWriter, GLPPredictor, GLPModelReader,
  GLPPosteriorOutputConstructor, GLPFactor, GLPWeights, GLPLossGradient}

import org.mitre.mandolin.glp.AbstractProcessor
import org.mitre.mandolin.transform.{ FeatureExtractor, FeatureImportance }
import org.mitre.mandolin.gm.{ Feature, NonUnitFeature }
import org.mitre.mandolin.util.LineParser
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import scala.reflect.ClassTag

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{ FileSystem, Path }


/**
 * @author wellner
 */
class DistributedProcessor(val numPartitions: Int) extends AbstractProcessor {
  def this() = this(0)

  //def processTrain(ev: GLPInstanceEvaluator)

  def processTrain(appSettings: GLPModelSettings) = {
    if (appSettings.trainFile.isEmpty) throw new RuntimeException("Training file required")
    val sc = AppConfig.getSparkContext(appSettings)
    val io = new IOAssistant(Some(sc))
    val components = getComponentsViaSettings(appSettings, io)
    val ev = components.evaluator
    val fe = components.featureExtractor
    val trainFile = appSettings.trainFile.get
    val network = ev.glp
    val optimizer: DistributedOptimizerEstimator[GLPFactor, GLPWeights] = DistributedGLPOptimizer.getDistributedOptimizer(sc, appSettings, network, ev)
    val lines =
      if (numPartitions > 0) sc.textFile(trainFile, numPartitions).coalesce(numPartitions, true)
      else sc.textFile(trainFile)
    val trainer = new Trainer[String, GLPFactor, GLPWeights](fe, optimizer, appSettings.storageLevel)
    val (w, _) = trainer.trainWeights(lines)
    val modelWriter = new GLPModelWriter(Some(sc))
    modelWriter.writeModel(io, appSettings.modelFile.get, w, components.labelAlphabet, ev, fe)
    w
  }

  def processDecode(appSettings: GLPModelSettings) = {
    if (appSettings.modelFile.isEmpty) throw new RuntimeException("Model file required as input in decoding mode")
    val sc = AppConfig.getSparkContext(appSettings)
    val io = new IOAssistant(Some(sc))
    val modelSpec = (new GLPModelReader(Some(sc))).readModel(appSettings.modelFile.get, io)
    val testLines = sc.textFile(appSettings.testFile.get, numPartitions).coalesce(numPartitions, true)
    val evaluator = modelSpec.evaluator
    val predictor = new GLPPredictor(evaluator.glp, true)
    val oc = new GLPPosteriorOutputConstructor()
    val decoder = new PosteriorDecoder(modelSpec.fe, predictor, oc)
    val os = io.getPrintWriterFor(appSettings.outputFile.get, false)
    val outputs = decoder.run(testLines, sc.broadcast(modelSpec.wts))
    writeOutputs(os, outputs.toLocalIterator, Some(modelSpec.la))
    os.close()
  }

  def processTrainTest(appSettings: GLPModelSettings) = {
    if (appSettings.trainFile.isEmpty) throw new RuntimeException("Training file required")    
    val sc = AppConfig.getSparkContext(appSettings)
    val io = new IOAssistant(Some(sc))
    val components = getComponentsViaSettings(appSettings, io)
    val ev = components.evaluator
    val fe = components.featureExtractor
    val pr = components.predictor

    val trainFile = appSettings.trainFile
    val optimizer = DistributedGLPOptimizer.getDistributedOptimizer(sc, appSettings, ev.glp, ev)
    val lines = sc.textFile(appSettings.trainFile.get, numPartitions).coalesce(numPartitions, true)
    val trainer = new Trainer(fe, optimizer)
    val testLines = sc.textFile(appSettings.testFile.get, numPartitions).coalesce(numPartitions, true)
    val trainTester =
      new TrainTester[String, GLPFactor, GLPWeights, GLPLossGradient, Int, DiscreteConfusion](trainer,
        pr, appSettings.numEpochs, appSettings.testFreq, sc, appSettings.progressFile)
    trainTester.trainAndTest(lines, testLines)
  }

  def processTrainDecode(appSettings: GLPModelSettings) = {
    val sc = AppConfig.getSparkContext(appSettings)
    val io = new IOAssistant(Some(sc))
    val components = getComponentsViaSettings(appSettings, io)
    val ev = components.evaluator
    val fe = components.featureExtractor
    val predictor = components.predictor
    val labelAlphabet = components.labelAlphabet
    val oc = components.outputConstructor
    val weights = processTrain(appSettings)
    val testLines = sc.textFile(appSettings.testFile.get, numPartitions).coalesce(numPartitions, true)
    val decoder = new PosteriorDecoder(fe, predictor, oc)
    val os = io.getPrintWriterFor(appSettings.outputFile.get, false)
    val outputs = decoder.run(testLines, sc.broadcast(weights))
    writeOutputs(os, outputs.toLocalIterator, Some(labelAlphabet))
    os.close()
  }
}

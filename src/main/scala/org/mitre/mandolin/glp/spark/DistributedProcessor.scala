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

import org.mitre.mandolin.glp.{AbstractProcessor, GLPModelSettings, GLPModelWriter, GLPPredictor, GLPModelReader,
  GLPPosteriorOutputConstructor, GLPFactor, GLPWeights, GLPLossGradient, GLPModelSpec, GLPInstanceEvaluator}

import org.mitre.mandolin.glp.AbstractProcessor
import org.mitre.mandolin.transform.{ FeatureExtractor, FeatureImportance }
import org.mitre.mandolin.gm.{ Feature, NonUnitFeature }
import org.mitre.mandolin.util.LineParser
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import scala.reflect.ClassTag
import com.twitter.chill.EmptyScalaKryoInstantiator

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{ FileSystem, Path }
import com.esotericsoftware.kryo.Kryo
import com.twitter.chill.AllScalaRegistrar
import org.apache.spark.serializer.KryoRegistrator
import org.mitre.mandolin.config.MandolinRegistrator


/**
 * Provides and registers a set of classes that will be serialized/deserialized
 * using Kryo for use within Spark.
 * @author wellner
 */
class MandolinKryoRegistrator extends KryoRegistrator with MandolinRegistrator {
  override def registerClasses(kryo: Kryo) = register(kryo)
    
}


/**
 * @author wellner
 */
class DistributedGLPModelWriter(sc: SparkContext) extends GLPModelWriter {
  val instantiator = new EmptyScalaKryoInstantiator

  val kryo = { 
      val k = new org.apache.spark.serializer.KryoSerializer(sc.getConf)      
      k.newKryo()
  }
  
  val registrator = new MandolinKryoRegistrator()
  registrator.registerClasses(kryo)

  def writeModel(weights: GLPWeights): Unit = {
    throw new RuntimeException("Intermediate model writing not implemented with GLPWeights")
  }

  def writeModel(io: IOAssistant, filePath: String, w: GLPWeights, la: Alphabet, ev: GLPInstanceEvaluator, fe: FeatureExtractor[String, GLPFactor]): Unit = {
    io.writeSerializedObject(kryo, filePath, GLPModelSpec(w, ev, la, fe))
  }
}

class DistributedGLPModelReader(sc: SparkContext) extends GLPModelReader {
  val instantiator = new EmptyScalaKryoInstantiator
  
  val kryo = { 
      val k = new org.apache.spark.serializer.KryoSerializer(sc.getConf)      
      k.newKryo()
  }

  val registrator = new MandolinKryoRegistrator()  
  registrator.registerClasses(kryo)

  /**
   * def readModel(f: java.io.File): GLPModelSpec = {
   * val is = new java.io.BufferedInputStream(new java.io.FileInputStream(f))
   * val kInput = new KInput(is)
   * val m = kryo.readObject(kInput, classOf[GLPModelSpec])
   * kInput.close()
   * is.close()
   * m
   * }
   */

  def readModel(f: String, io: IOAssistant): GLPModelSpec = {
    io.readSerializedObject(kryo, f, classOf[GLPModelSpec]).asInstanceOf[GLPModelSpec]
  }
}



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
    val modelWriter = new DistributedGLPModelWriter(sc)
    modelWriter.writeModel(io, appSettings.modelFile.get, w, components.labelAlphabet, ev, fe)
    w
  }

  def processDecode(appSettings: GLPModelSettings) = {
    if (appSettings.modelFile.isEmpty) throw new RuntimeException("Model file required as input in decoding mode")
    val sc = AppConfig.getSparkContext(appSettings)
    val io = new IOAssistant(Some(sc))
    val modelSpec = (new DistributedGLPModelReader(sc)).readModel(appSettings.modelFile.get, io)
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

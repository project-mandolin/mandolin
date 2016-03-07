package org.mitre.mandolin.glp.spark
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */


import org.mitre.mandolin.util.{ StdAlphabet, PrescaledAlphabet, RandomAlphabet, Alphabet, IdentityAlphabet, AlphabetWithUnitScaling, IOAssistant }
import org.mitre.mandolin.util.spark.SparkIOAssistant
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
  GLPPosteriorOutputConstructor, GLPFactor, GLPWeights, GLPLossGradient, GLPModelSpec, GLPInstanceEvaluator, ANNetwork}
import org.mitre.mandolin.transform.{ FeatureExtractor, FeatureImportance }
import org.mitre.mandolin.gm.{ Feature, NonUnitFeature }
import org.mitre.mandolin.util.LineParser
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel
import scala.reflect.ClassTag
import com.twitter.chill.EmptyScalaKryoInstantiator

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{ FileSystem, Path }
import com.esotericsoftware.kryo.Kryo
import com.twitter.chill.AllScalaRegistrar
import org.apache.spark.serializer.KryoRegistrator
import org.mitre.mandolin.config.MandolinRegistrator
import org.mitre.mandolin.config.AppSettings


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

  def writeModel(io: IOAssistant, filePath: String, w: GLPWeights, la: Alphabet, ann: ANNetwork, fe: FeatureExtractor[String, GLPFactor]): Unit = {
    io.writeSerializedObject(kryo, filePath, GLPModelSpec(w, ann, la, fe))
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
  // add in additional Spark data structures to register
  kryo.register(classOf[org.apache.spark.util.Vector])

  def readModel(f: String, io: IOAssistant): GLPModelSpec = {
    io.readSerializedObject(kryo, f, classOf[GLPModelSpec]).asInstanceOf[GLPModelSpec]
  }
}

/**
 * @author wellner
 */
object AppConfig {    
 
  import scala.collection.JavaConversions._
  
  /**
   * Creates a Spark Configuration object and adds application-specific Spark settings
   * to the configuration. This allows Mandolin application settings and Spark settings
   * to be specified in a unified way via the Typesafe configuration framework.
   * @param appSettings Application settings object
   */
  private def getSparkConf(appSettings: AppSettings) = {
    val unwrappedA = appSettings.config.getConfig("spark").entrySet().toList
    val unwrapped = unwrappedA map {entry => (entry.getKey(), entry.getValue.unwrapped().toString) }
    
    val conf = new SparkConf()
        .setAppName(appSettings.name)
        //.setJars(appSettings.appJar)

    unwrapped foreach {case (k,v) =>
      println("Setting conf: " + ("spark."+k) + " to " + v)
      conf.set(("spark."+k),v.toString())} 
    conf
  }

  /**
   * Instantiate a new SparkContext. The Application-wide configuration
   * object is provided here and all Spark-specific config options are
   * passed to Spark
   * @param appSettings Application settings object
   */
  def getSparkContext(appSettings:AppSettings) = {
    val conf = getSparkConf(appSettings)
    val context = new SparkContext(conf)
    context
  }
  
}



/**
 * @author wellner
 */
class DistributedProcessor(val numPartitions: Int) extends AbstractProcessor {
  def this() = this(0)
  
  def getStorageLevel(appSettings: GLPModelSettings) = appSettings.storage match {
    case "mem_ser_only" => StorageLevel.MEMORY_ONLY_SER
    case "disk_only" => StorageLevel.DISK_ONLY
    case "mem_and_disk" => StorageLevel.MEMORY_AND_DISK
    case "mem_and_disk_ser" => StorageLevel.MEMORY_AND_DISK_SER
    case _ => StorageLevel.MEMORY_ONLY
  }

  //def processTrain(ev: GLPInstanceEvaluator)

  def processTrain(appSettings: GLPModelSettings) = {
    if (appSettings.trainFile.isEmpty) throw new RuntimeException("Training file required")
    val sc = AppConfig.getSparkContext(appSettings)
    val io = new SparkIOAssistant(sc)
    val components = getComponentsViaSettings(appSettings, io)
    val fe = components.featureExtractor
    val trainFile = appSettings.trainFile.get
    val network = components.ann
    val optimizer: DistributedOptimizerEstimator[GLPFactor, GLPWeights] = DistributedGLPOptimizer.getDistributedOptimizer(sc, appSettings, network)
    val lines =
      if (numPartitions > 0) sc.textFile(trainFile, numPartitions).coalesce(numPartitions, true)
      else sc.textFile(trainFile)
    val trainer = new Trainer[String, GLPFactor, GLPWeights](fe, optimizer, getStorageLevel(appSettings))
    val (w, _) = trainer.trainWeights(lines)
    val modelWriter = new DistributedGLPModelWriter(sc)
    modelWriter.writeModel(io, appSettings.modelFile.get, w, components.labelAlphabet, network, fe)
    w
  }

  def processDecode(appSettings: GLPModelSettings) = {
    if (appSettings.modelFile.isEmpty) throw new RuntimeException("Model file required as input in decoding mode")
    val sc = AppConfig.getSparkContext(appSettings)
    val io = new SparkIOAssistant(sc)
    val modelSpec = (new DistributedGLPModelReader(sc)).readModel(appSettings.modelFile.get, io)
    val testLines = sc.textFile(appSettings.testFile.get, numPartitions).coalesce(numPartitions, true)
    val predictor = new GLPPredictor(modelSpec.ann, true)
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
    val io = new SparkIOAssistant(sc)
    val components = getComponentsViaSettings(appSettings, io)
    val fe = components.featureExtractor
    val pr = components.predictor
    val trainFile = appSettings.trainFile
    val optimizer = DistributedGLPOptimizer.getDistributedOptimizer(sc, appSettings, components.ann)
    val lines = sc.textFile(appSettings.trainFile.get, numPartitions).coalesce(numPartitions, true)
    val trainer = new Trainer(fe, optimizer)
    val testLines = sc.textFile(appSettings.testFile.get, numPartitions).coalesce(numPartitions, true)
    val trainTester =
      new TrainTester[String, GLPFactor, GLPWeights, GLPLossGradient, Int, DiscreteConfusion](trainer,
        pr, appSettings.numEpochs, appSettings.testFreq, sc, appSettings.progressFile, getStorageLevel(appSettings))
    trainTester.trainAndTest(lines, testLines)
  }

  def processTrainDecode(appSettings: GLPModelSettings) = {
    val sc = AppConfig.getSparkContext(appSettings)
    val io = new SparkIOAssistant(sc)
    val components = getComponentsViaSettings(appSettings, io)
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

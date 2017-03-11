package org.mitre.mandolin.config
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import com.typesafe.config.{Config, ConfigFactory, ConfigRenderOptions}
import net.ceedubs.ficus.Ficus._

/**
 * This top-level abstract class holds general Spark configuration
 * settings not specific to any particular type of Mandolin application.
 * Note that many of these set Spark configuration options, more detail
 * of which can be found 
 * <a href="http://spark.apache.org/docs/latest/configuration.html">here</a>
 * @param args - command-line arguments as a Seq of String objects
 */ 
abstract class GeneralSettings(args: Seq[String]) {
  import scala.collection.JavaConverters._
  import org.apache.log4j.{ Level, Logger, LogManager }
  
  protected val commandOptions = new ConfigGeneratedCommandOptions(args.toSeq)        
  lazy val config = commandOptions.finalConfig
  
  if (commandOptions.displayDefaults()) {
      val configRenderer = ConfigRenderOptions.defaults().setOriginComments(false)
      println("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      println("++++++++++ Mandolin Configuration ++++++++++++++++++++++++++++++++")
      println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      println(config.getConfig("mandolin").root().render(configRenderer))
      println("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      System.exit(0)
  }
  
  def getConfigValue(path: String) = {
    asStr(path)
  }
      
  protected def asStrOpt(key: String) = 
    try {
      config.getString(key) match { case "" | "null" | "NULL" => None case s => Some(s) }    
    } catch {case e: Throwable => None}

  protected def asIntOpt(key: String) : Option[Int] = {
    try {
      Some(config.getInt(key))
    } catch {case e: Throwable => None}    
  }
  
  protected def asFloatOpt(key: String) : Option[Float] = {
    try {
      Some(config.getDouble(key).toFloat)
    } catch {case e: Throwable => None}
  }
  
  protected def asStr(key: String) = {
    config.getString(key)            
  }
  
  protected def asInt(key: String) = { 
    config.getInt(key)            
  }
  
  protected def asDouble(key: String) = { 
    config.getDouble(key)        
  }
  
  protected def asFloat(key: String) = { 
    config.getDouble(key).toFloat        
  }
  
  protected def asBoolean(key: String) = {
    config.getBoolean(key)        
  }

  /*  
  protected def mapStorage(s: String) = s match {
    case "mem_only"          => StorageLevel.MEMORY_ONLY
    case "mem_and_disk"      => StorageLevel.MEMORY_AND_DISK
    case "disk_only"         => StorageLevel.DISK_ONLY
    case "mem_and_disk_ser"  => StorageLevel.MEMORY_AND_DISK_SER
    case "mem_only_ser"      => StorageLevel.MEMORY_ONLY_SER
    case "none"              => StorageLevel.NONE
    case _                   => StorageLevel.MEMORY_ONLY
  }
  */
  
  protected def mapLogLevel(s: String) = s match {
    case "INFO"  => Level.INFO
    case "DEBUG" => Level.DEBUG
    case "ERROR" => Level.ERROR
    case "OFF"   => Level.OFF
    case "FATAL" => Level.FATAL
    case "ALL"   => Level.ALL
    case _       => Level.OFF
  }
  

  /** Number of Spark partitions/slices to use for the application */
  val numPartitions   = asInt("mandolin.partitions")
  
  /** Spark storage level for application */
  //val storageLevel    = mapStorage(asStr("spark.storage"))
  
  /** General log-level for Spark and Mandolin */
  val logLevel = {
    val ll = asStrOpt("mandolin.log-level")
    ll foreach {l => System.setProperty("log4j.logLevel", l)}
    ll
  }
  
  /** Logging config file - set as system property immediately */
  val logConfigFile = {
    val cf = asStrOpt("mandolin.log-config-file")
    cf foreach {f => System.setProperty("log4j.configuration", ("file:" + f))}
    cf
  }
  
  /** Log4j output file. Overrides log4j configuration if set */
  val logOutputFile = {    
    val lf = asStrOpt("mandolin.log-file")
    lf foreach {l => System.setProperty("log4j.logOutFile", l)}
    lf
  }
  
  /*
  protected val getAppJar : Option[String] = asStrOpt("mandolin.jar-file")  
  protected lazy val defaultAppJar : Seq[String] = {
    SparkContext.jarOfClass(this.getClass) match {
      case Some(f) => Seq(SparkContext.jarOfClass(this.getClass).get)
      case None => Seq.empty
    }
   }
  
  /** The application jar itself - if not provided it is inferred */
  def appJar : Seq[String] = getAppJar match {case None => defaultAppJar case Some(j) => Seq(j) }
  */
}

/**
 * Mandolin application settings
 * @param args - command-line args
 */ 
abstract class AppSettings(args: Seq[String]) extends GeneralSettings(args) {
  /** Name for the app */
  val name             = asStr("mandolin.name")
  /** Mode for application (train|decode|train-test|train-decode) */
  val appMode          = asStr("mandolin.mode")
  val storage          = asStr("mandolin.spark.storage")
}



/**
 * Settings specific to all Mandolin learners
 * @param args - command-line args
 */ 
abstract class LearnerSettings(args: Seq[String]) extends AppSettings(args) {   
  
  
  
  val numFeatures      = asInt("mandolin.trainer.num-hash-features")
  val trainFile        = asStrOpt("mandolin.trainer.train-file")
  val testFile         = asStrOpt("mandolin.trainer.test-file")
  val testFreq         = asInt("mandolin.trainer.eval-freq")
  val testPartitions   = asInt("mandolin.trainer.test-partitions")
  val modelFile        = asStrOpt("mandolin.trainer.model-file")

  val numEpochs        = asInt("mandolin.trainer.num-epochs")
  val numSubEpochs     = asInt("mandolin.trainer.num-subepochs")
  val detailsFile      = asStrOpt("mandolin.trainer.detail-file")
  val progressFile     = asStrOpt("mandolin.trainer.progress-file")
  //val lossFunction     = asStrOpt("mandolin.trainer.loss-function")
  val labelFile        = asStrOpt("mandolin.trainer.label-file")
  // unused?
  val modelType        = asStrOpt("mandolin.trainer.model-type")
  val ensureSparse     = asBoolean("mandolin.trainer.ensure-sparse")
  val useRandom        = asBoolean("mandolin.trainer.use-random-features")
  val printFeatureFile = asStrOpt("mandolin.trainer.print-feature-file")
  val filterFeaturesMI = asInt("mandolin.trainer.max-features-mi")
  
  // these should move to model specification as they are specific to loss functions
  val coef1            = asDouble("mandolin.trainer.coef1")
  val qval             = asDouble("mandolin.trainer.qval")
  

  val oversampleRatio  = asDouble("mandolin.trainer.oversample")
  val denseVectorSize  = asInt("mandolin.trainer.dense-vector-size")
  val scaleInputs      = asBoolean("mandolin.trainer.scale-inputs")
  val composeStrategy  = asStr("mandolin.trainer.updater-compose-strategy")
  val maxNorm          = asBoolean("mandolin.trainer.max-norm")
  val denseOutputFile  = asStrOpt("mandolin.trainer.dense-output-file") // output vectors in dense format
  val numThreads       =     asInt("mandolin.trainer.threads")
  val skipProb : Double = asFloat("mandolin.trainer.skip-probability")
  val miniBatchSize    =     asInt("mandolin.trainer.mini-batch-size")
  val synchronous      = asBoolean("mandolin.trainer.synchronous")
  val sgdLambda        =  asFloat("mandolin.trainer.optimizer.lambda")  
  val epsilon          =  asFloat("mandolin.trainer.optimizer.epsilon")
  val rho              =  asFloat("mandolin.trainer.optimizer.rho")
  val method           =     asStr("mandolin.trainer.optimizer.method")
  val initialLearnRate =  asFloat("mandolin.trainer.optimizer.initial-learning-rate")
  
}

abstract class GeneralLearnerSettings[S <: GeneralLearnerSettings[S]](args: Seq[String]) extends LearnerSettings(args) {
  def withSets(avs: Seq[(String, Any)]) : S
}

/*
 * Trait holds options specific to batch learner
 */
trait BatchLearnerSettings extends AppSettings {
  val densityRatio = asDouble("mandolin.trainer.density-ratio")  
}



trait DeepNetSettings extends AppSettings {
  val netspec       = config.as[List[Map[String,String]]]("mandolin.trainer.specification") 
}


/**
 * Settings for all Mandolin decoders
 * @param args - command-line arguments
 */ 
trait DecoderSettings extends AppSettings {  
  val decoderInputFile  = asStrOpt("mandolin.decoder.input-file")
  val outputFile        = asStrOpt("mandolin.decoder.output-file")
  val inputModelFile    = asStrOpt("mandolin.decoder.model-file")
}

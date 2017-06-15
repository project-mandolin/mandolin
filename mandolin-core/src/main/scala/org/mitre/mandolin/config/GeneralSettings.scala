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
abstract class GeneralSettings(confOptions: Option[ConfigGeneratedCommandOptions], conf: Option[Config]) {
  import scala.collection.JavaConverters._
  import org.apache.log4j.{ Level, Logger, LogManager }
  
  def this(c: Config) = this(None, Some(c))
  def this(args: Seq[String]) = this(Some(new ConfigGeneratedCommandOptions(args)), None)
  
  lazy val config = conf.getOrElse(getConfig)
  
  def getAttVals(s: String) : (String, Any) = {
    val s1 = s.split('=')
    val k = s1(0)
    val v = s1(1)
    (k,v)
  }
  
  def getConfig = {
    val options = confOptions.get
    val overs = try { options.overrides() } catch {  case _:Throwable => Nil }
    val conf1 = options.finalConfig    
    val nc = overs.foldLeft(conf1){case (ac, s) => 
      val (k,v) = getAttVals(s)
      ac.withValue(k, com.typesafe.config.ConfigValueFactory.fromAnyRef(v))
    }
    nc.resolve()
  }
  
  //protected val commandOptions = new ConfigGeneratedCommandOptions(args.toSeq)        
  //lazy val config = commandOptions.finalConfig

  /*
  if (commandOptions.displayDefaults()) {
      val configRenderer = ConfigRenderOptions.defaults().setOriginComments(false)
      println("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      println("++++++++++ Mandolin Configuration ++++++++++++++++++++++++++++++++")
      println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      println(config.getConfig("mandolin").root().render(configRenderer))
      println("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      System.exit(0)
  }
  */
  
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
    try config.getBoolean(key) catch {case _: Throwable => false}        
  }
  
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
}

/**
 * Mandolin application settings
 * @param args - command-line args
 */ 
abstract class AppSettings(_confOptions: Option[ConfigGeneratedCommandOptions], _conf: Option[Config]) extends GeneralSettings(_confOptions, _conf) {
  /** Name for the app */
  val name             = asStr("mandolin.name")
  /** Mode for application (train|decode|train-test|train-decode) */
  val appMode          = asStr("mandolin.mode")
  val storage          = asStr("mandolin.spark.storage") // TODO
}



/**
 * Settings specific to all Mandolin learners
 * @param args - command-line args
 */ 
abstract class MandolinMLPSettings(_confOptions: Option[ConfigGeneratedCommandOptions], _conf: Option[Config]) extends AppSettings(_confOptions, _conf) {


  val numFeatures      = asInt("mandolin.mmlp.num-hash-features")
  val trainFile        = asStrOpt("mandolin.trainer.train-file")
  val testFile         = asStrOpt("mandolin.trainer.test-file")
  val testFreq         = asInt("mandolin.mmlp.eval-freq")
  val testPartitions   = asInt("mandolin.mmlp.test-partitions")
  val modelFile        = asStrOpt("mandolin.mmlp.model-file")

  val numEpochs        = asInt("mandolin.mmlp.num-epochs")
  val numSubEpochs     = asInt("mandolin.mmlp.num-subepochs")
  val detailsFile      = asStrOpt("mandolin.mmlp.detail-file")
  val progressFile     = asStrOpt("mandolin.mmlp.progress-file")
  //val lossFunction     = asStrOpt("mandolin.mmlp.loss-function")
  val labelFile        = asStrOpt("mandolin.mmlp.label-file")
  // unused?
  val modelType        = asStrOpt("mandolin.mmlp.model-type")
  val ensureSparse     = asBoolean("mandolin.mmlp.ensure-sparse")
  val useRandom        = asBoolean("mandolin.mmlp.use-random-features")
  val printFeatureFile = asStrOpt("mandolin.mmlp.print-feature-file")
  val filterFeaturesMI = asInt("mandolin.mmlp.max-features-mi")

  // these should move to model specification as they are specific to loss functions
  val coef1            = asDouble("mandolin.mmlp.coef1")
  val qval             = asDouble("mandolin.mmlp.qval")

// TODO are these unique to mmlp?
  val oversampleRatio  = asDouble("mandolin.mmlp.oversample")
  val denseVectorSize  = asInt("mandolin.mmlp.dense-vector-size")
  val scaleInputs      = asBoolean("mandolin.mmlp.scale-inputs")
  val composeStrategy  = asStr("mandolin.mmlp.updater-compose-strategy")
  val maxNorm          = asBoolean("mandolin.mmlp.max-norm")
  val denseOutputFile  = asStrOpt("mandolin.mmlp.dense-output-file") // output vectors in dense format
  val numThreads       =     asInt("mandolin.mmlp.threads")
  val skipProb : Double = asFloat("mandolin.mmlp.skip-probability")
  val miniBatchSize    =     asInt("mandolin.mmlp.mini-batch-size")
  val synchronous      = asBoolean("mandolin.mmlp.synchronous")
  val sgdLambda        =  asFloat("mandolin.mmlp.optimizer.lambda")
  val epsilon          =  asFloat("mandolin.mmlp.optimizer.epsilon")
  val rho              =  asFloat("mandolin.mmlp.optimizer.rho")
  val method           =     asStr("mandolin.mmlp.optimizer.method")
  val initialLearnRate =  asFloat("mandolin.mmlp.optimizer.initial-learning-rate")

}

abstract class GeneralLearnerSettings[S <: GeneralLearnerSettings[S]](_confOptions: Option[ConfigGeneratedCommandOptions], _conf: Option[Config])
extends MandolinMLPSettings(_confOptions, _conf) {
  def withSets(avs: Seq[(String, Any)]) : S
}

/*
 * Trait holds options specific to batch learner
 */
trait BatchLearnerSettings extends AppSettings {
  val densityRatio = asDouble("mandolin.mmlp.density-ratio")
}



trait DeepNetSettings extends AppSettings {
  
  def mapSpecToList(conf: Map[String, Map[String, String]]) = {
    val layerNames = conf.keySet
    var prevName = ""
    val nextMap = layerNames.toSet.foldLeft(Map():Map[String,String]){case (ac,v) =>
      val cc = conf(v)
        try {
        val inLayer = cc("data")
        ac + (inLayer -> v)
        } catch {case _:Throwable =>
          prevName = v  // this is the name for the input layer (as it has no "data" field")
          ac}
      }      
    var building = true    
    val buf = new collection.mutable.ArrayBuffer[String]    
    buf append prevName // add input layer name first
    while (building) {
      val current = nextMap.get(prevName)
      current match {case Some(c) => buf append c; prevName = c case None => building = false}
      }
    buf.toList map {n => conf(n)} // back out as an ordered list      
  }

  val netspec       = try { config.as[List[Map[String,String]]]("mandolin.trainer.specification") } catch {case _: Throwable =>
    Nil
    } 
  val netspecConfig : Option[Map[String, Map[String, String]]] =  
    try { Some(config.as[Map[String, Map[String,String]]]("mandolin.trainer.specification")) } 
    catch {case _: Throwable => None }
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

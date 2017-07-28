package org.mitre.mandolin.config
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import com.typesafe.config.{Config, ConfigFactory, ConfigRenderOptions}
import net.ceedubs.ficus.Ficus._

// TODO update parameter

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
  import org.apache.log4j.Level
  
  def this(c: Config) = this(None, Some(c))
  def this(args: Seq[String]) = this(Some(new ConfigGeneratedCommandOptions(args)), None)
  
  lazy val config: Config = conf.getOrElse(getConfig)
  
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
  

  /** Number of Spark partitions to use for the application */
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

// TODO update parameter

/**
 * Mandolin application settings
 * @param args - command-line args
 */ 
abstract class AppSettings[S <: AppSettings[S]](_confOptions: Option[ConfigGeneratedCommandOptions], _conf: Option[Config]) extends GeneralSettings(_confOptions, _conf) {
  /** Name for the app */
  val name             = asStr("mandolin.name")
  /** Mode for application (train|predict|model-selection) */
  val appMode          = asStr("mandolin.mode")
  val storage          = asStr("mandolin.spark.storage")
  val driverBlock      = asStrOpt("mandolin.driver").getOrElse("mmlp")
  def withSets(avs: Seq[(String, Any)]) : S
}

class InitializeSettings(_confOptions: Option[ConfigGeneratedCommandOptions], _conf: Option[Config]) extends AppSettings[InitializeSettings](_confOptions, _conf) with Serializable {

  import scala.collection.JavaConversions._
  
  def this(str: String) = this(None, Some(com.typesafe.config.ConfigFactory.parseString(str)))
  def this(args: Array[String]) = this(Some(new ConfigGeneratedCommandOptions(args)), None)
  
  /**
    * Returns a new settings object with the sequence of tuple arguments values set accordingly
    */
  override def withSets(avs: Seq[(String, Any)]): InitializeSettings = {
    val nc = avs.foldLeft(this.config) { case (ac, (v1, v2)) =>
      v2 match {
        case v2: List[_] =>
          if (v2 != null) ac.withValue(v1, com.typesafe.config.ConfigValueFactory.fromIterable(v2)) else ac
        case v2: Any =>
          ac.withValue(v1, com.typesafe.config.ConfigValueFactory.fromAnyRef(v2))
      }
    }
    new InitializeSettings(None, Some(nc))
  }
}

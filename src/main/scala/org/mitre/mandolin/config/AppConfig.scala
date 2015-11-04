package org.mitre.mandolin.config
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import com.typesafe.config.{ Config, ConfigFactory, ConfigObject, ConfigList }
import org.apache.spark.{ SparkConf, SparkContext }
import scala.collection.JavaConversions._

/**
 * @author wellner
 */
object AppConfig {    
 
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
        .setJars(appSettings.appJar)

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


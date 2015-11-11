package org.mitre.mandolin.config
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import scala.util.Properties
import org.rogach.scallop.{ScallopConf, ScallopOption}
import com.typesafe.config.{ Config, ConfigObject, ConfigValueType, ConfigList, ConfigValue, ConfigRenderOptions, ConfigFactory, ConfigValueFactory }

/**
 * Takes a sequence of arguments from the command line and adds to the default configuration
 * object constructed from the src/main/resources/reference.conf and src/main/resource/application.conf
 * files.
 * @param args sequence of command line argument strings
 */
class ConfigGeneratedCommandOptions(args: Seq[String]) extends ScallopConf(args) {
  import scala.collection.JavaConverters._

  /** Runtime Mandolin configuration file */
  val configFile = opt[String]("conf", required = false, descr = "Configuration file")

  /** Display all configuration options */
  val displayDefaults = opt[Boolean]("X-display", default = Some(false), descr = "Display all defaults based on provided config(s); then exit program")

  val overrides = trailArg[List[String]](required=false)

  def addPathWithValue(ref: Config, c: Config, p: String, v: String) = {
    //if (ref.hasPath(p)) {
    if (true) {
      c.withValue(p, ConfigValueFactory.fromAnyRef(v))
    } else {
      throw new RuntimeException("Invalid config argument: " + p)
    }
  }

  def applyArgs(config: Config, commandConfig: Config) = {
    try {
      val overs = overrides()
      if (overs.length > 0) {
        if (overs(0).contains('=')) { // first arg has an '=' sign, assume att=val format for args
          overs.foldLeft(commandConfig) {
            case (c, kv) =>
              val kvp = kv.split('=')
              val k = kvp(0)
              val v = kvp(1)
              addPathWithValue(config, c, k, v)
          }
        } else { // first 'arg' doesn't have equals so assume args were space-separated
          if (overs.length > 1)
            overs.sliding(2, 1).foldLeft(commandConfig) { case (c, kv) => addPathWithValue(config, c, kv(0), kv(1)) }
          else throw new RuntimeException("Invalid config overrides: " + overs)
        }
      } else config
    } catch { case _: Throwable => config }
  }

  lazy val finalConfig = {
    val fileAugConfig =
      configFile.get match {
        case Some(f) => ConfigFactory.parseFile(new java.io.File(f)).withFallback(ConfigFactory.load())
        case None => ConfigFactory.load()
    }
    val conf = applyArgs(fileAugConfig, ConfigFactory.empty())
    val opts = com.typesafe.config.ConfigResolveOptions.defaults().setAllowUnresolved(true)
    val updated = fileAugConfig.resolveWith(conf, opts)
    applyArgs(fileAugConfig, updated.withFallback(fileAugConfig).resolve()).resolve()
  }

  def getOptValAsString(s: String): Option[String] = {
    try { Some(finalConfig.getString(s)) } catch { case _: Throwable => None }
  }

}

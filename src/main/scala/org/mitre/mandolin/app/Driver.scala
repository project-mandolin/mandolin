package org.mitre.mandolin.app
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */
import org.mitre.mandolin.config.AppSettings

trait AppMain {
  def main(args: Array[String]) : Unit
}

case class MissingMainException(s: String) extends Exception

/**
 * Primary driver class for Mandolin applications. This class should be called
 * from the command-line or Spark-submit script. The application-specific class
 * is specified through Mandolin configuration.
 * @author wellner
 */
object Driver extends org.apache.spark.Logging with AppMain {
  import scala.reflect.runtime.universe

  def main(args: Array[String]): Unit = {
    val tmpSettings = new AppSettings(args) {} // create settings from specified configuration file to find main class to load
    val driverClassName = tmpSettings.appDriverClass
    try {
      val runtimeMirror = universe.runtimeMirror(getClass.getClassLoader())
      val module = runtimeMirror.staticModule(driverClassName)
      val obj = runtimeMirror.reflectModule(module)
      obj.instance match {
        case inst: AppMain => inst.main(args)
        case _ => throw new MissingMainException("Specified driver class does not implement AppMain trait.") 
      } 
    } catch {
      case m: MissingMainException =>
        logError("Driver class name: " + driverClassName + " isn't found.  Check whether driver class name is valid.")
        throw m
      case e: Exception =>        
        throw e
    }
  }

}

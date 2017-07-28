package org.mitre.mandolin.app

import org.mitre.mandolin.config.InitializeSettings
import org.slf4j.LoggerFactory
import scala.reflect.runtime.universe


trait AppMain {
  def main(args: Array[String]): Unit
}

case class MissingMainException(s: String) extends Exception

/**
  * Created by jkraunelis on 6/15/17.
  */
object Mandolin extends AppMain with org.mitre.mandolin.config.LogInit {

  val logger = LoggerFactory.getLogger(this.getClass)

  def main(args: Array[String]): Unit = {
    val tmpSettings = new InitializeSettings(args) // create settings from specified configuration file to find main class to load

    val driverClassName =
      tmpSettings.driverBlock match {
        case "mx" => "org.mitre.mandolin.mx.local.MxMain"
        case "xg" => "org.mitre.mandolin.xg.XGMain"
        case _ => "org.mitre.mandolin.glp.local.MandolinWithoutSpark"
      }
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
        logger.error("Driver class name: " + driverClassName + " isn't found.  Check whether driver class name is valid.")
        throw m
      case e: Exception =>
        throw e
    }
  }
}

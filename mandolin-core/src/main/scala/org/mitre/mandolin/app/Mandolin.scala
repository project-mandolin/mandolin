package org.mitre.mandolin.app

import org.mitre.mandolin.config.AppSettings

import scala.reflect.runtime.universe

/**
  * Created by jkraunelis on 6/15/17.
  */
object Mandolin {
  def main(args: Array[String]): Unit = {
    /*  val tmpSettings = new AppSettings(args) // create settings from specified configuration file to find main class to load

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
  */
  }

}

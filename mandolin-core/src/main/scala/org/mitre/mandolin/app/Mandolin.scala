package org.mitre.mandolin.app

import com.typesafe.config.Config
import org.mitre.mandolin.config.{AppSettings, ConfigGeneratedCommandOptions, LogInit}

import scala.reflect.runtime.universe

class TmpAppSettings(_confOptions: Option[ConfigGeneratedCommandOptions], _conf: Option[Config]) extends AppSettings[TmpAppSettings](_confOptions, _conf) {
  override def withSets(avs: Seq[(String, Any)]): TmpAppSettings = this

  val appDriverClass = asStrOpt("mandolin.driver-class")
}

trait AppMain extends LogInit {
  def main(args: Array[String]): Unit
}

case class MissingMainException(s: String) extends Exception

/**
  * Created by jkraunelis on 6/15/17.
  */
object Mandolin extends AppMain {
  def main(args: Array[String]): Unit = {

    val tmpSettings = new TmpAppSettings(Some(new ConfigGeneratedCommandOptions(args)), None)
    val driverClassName = tmpSettings.appDriverClass.get
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
        println("Driver class name: " + driverClassName + " isn't found.  Check whether driver class name is valid.")
        throw m
      case e: Exception =>
        throw e
    }
  }
}

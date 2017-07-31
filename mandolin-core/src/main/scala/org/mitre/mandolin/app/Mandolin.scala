package org.mitre.mandolin.app

import org.mitre.mandolin.config.InitializeSettings
import org.slf4j.LoggerFactory
import scala.reflect.runtime.universe
import scala.collection.JavaConverters._

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
    // Spark 2.1 sets at least these 8 environment variables:
    // SPARK_SCALA_VERSION, SPARK_HOME, SPARK_ENV_LOADED, SPARK_LOCAL_DIRS, SPARK_MASTER_WEBUI_PORT, SPARK_WORKER_MEMORY, SPARK_SUBMIT_OPTS, SPARK_MASTER_IP
    val isDistributed = System.getenv().asScala.keys.filter(_.startsWith("SPARK")).size >= 8

    val driverClassName =
      if (tmpSettings.appMode != "model-selection") {
        // train/predict modes
        tmpSettings.driverBlock match {
          case "mx" => "org.mitre.mandolin.mx.standalone.MxMain"
          case "xg" => "org.mitre.mandolin.xg.XGMain"
          case _ =>
            if (isDistributed) "org.mitre.mandolin.mlp.spark.MandolinMain" else "org.mitre.mandolin.mlp.standalone.MandolinMain"
        }
      } else {
        // model selection mode
        tmpSettings.driverBlock match {
          case "mx" => if (isDistributed) "org.mitre.mandolin.mselect.SparkMxModelSelectionDriver" else "org.mitre.mandolin.mselect.MxLocalModelSelector"
          case "xg" => "org.mitre.mandolin.mselect.XGLocalModelSelector"
          case _ => if(isDistributed) "org.mitre.mandolin.mselect.SparkModelSelectionDriver" else "org.mitre.mandolin.mselect.standalone.ModelSelector"

        }
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

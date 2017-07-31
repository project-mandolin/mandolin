package org.mitre.mandolin.mlp.standalone

import org.mitre.mandolin.app.AppMain
import org.mitre.mandolin.config.LogInit
import org.mitre.mandolin.mlp.MandolinMLPSettings

/**
 * @author wellner
 */
object MandolinMain extends LogInit with AppMain {

  def main(args: Array[String]): Unit = {
    val appSettings = new MandolinMLPSettings(args)
    val mode = appSettings.appMode
    val numPartitions = appSettings.numPartitions
    val localProcessor = new Processor
    mode match {
      case "train"             => localProcessor.processTrain(appSettings)
      case "decode"            => localProcessor.processDecode(appSettings)
      case "train-test"        => localProcessor.processTrainTest(appSettings)
      case "train-decode"      => localProcessor.processTrainDecode(appSettings)
      case "train-decode-dirs" => localProcessor.processTrainTestDirectories(appSettings)
    }
    System.exit(0)
  }
}

package org.mitre.mandolin.glp.local

import org.mitre.mandolin.glp.MandolinMLPSettings

/**
 * @author wellner
 */
object MandolinWithoutSpark extends org.mitre.mandolin.config.LogInit {

  def main(args: Array[String]): Unit = {
    val appSettings = new MandolinMLPSettings(args)
    val mode = appSettings.appMode
    val numPartitions = appSettings.numPartitions
    val localProcessor = new LocalProcessor
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

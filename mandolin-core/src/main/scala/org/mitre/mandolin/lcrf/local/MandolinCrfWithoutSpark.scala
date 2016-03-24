package org.mitre.mandolin.lcrf.local

class MandolinCrfWithoutSpark {

}

object MandolinCrfWithoutSpark {

  def main(args: Array[String]): Unit = {
    val appSettings = new CrfModelSettings(args)
    val mode = appSettings.appMode
    val numPartitions = appSettings.numPartitions
    val localProcessor = new LocalCrfProcessor
    mode match {
      case "train"             => localProcessor.processTrain(appSettings)
      case "decode"            => localProcessor.processDecode(appSettings)
      case "train-test"        => localProcessor.processTrainTest(appSettings)
      //case "train-decode"      => localProcessor.processTrainDecode(appSettings)
      //case "train-decode-dirs" => localProcessor.processTrainTestDirectories(appSettings)
    }
    System.exit(0)
  }
}

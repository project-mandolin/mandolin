package org.mitre.mandolin.glp.spark

import org.mitre.mandolin.glp.GLPModelSettings
import org.mitre.mandolin.glp.local.LocalProcessor

/**
 * @author wellner
 */
object Mandolin extends org.mitre.mandolin.app.AppMain {

  def main(args: Array[String]): Unit = {
    val appSettings = new GLPModelSettings(args)
    val local_p = appSettings.appLocalOnly
    val mode = appSettings.appMode
    val numPartitions = appSettings.numPartitions
    if (local_p) {
      val localProcessor = new LocalProcessor
      mode match {
        case "train" => localProcessor.processTrain(appSettings)
        case "decode" => localProcessor.processDecode(appSettings)
        case "train-test" => localProcessor.processTrainTest(appSettings)
        case "train-decode" => localProcessor.processTrainDecode(appSettings)
        case "train-decode-dirs" => localProcessor.processTrainTestDirectories(appSettings)
      }
    } else {
      val distProcessor = new DistributedProcessor(numPartitions)
      mode match {
        case "train" => distProcessor.processTrain(appSettings)
        case "decode" => distProcessor.processDecode(appSettings)
        case "train-test" => distProcessor.processTrainTest(appSettings)
        case "train-decode" => distProcessor.processTrainDecode(appSettings)
      }
    }
    System.exit(0)
  }
}

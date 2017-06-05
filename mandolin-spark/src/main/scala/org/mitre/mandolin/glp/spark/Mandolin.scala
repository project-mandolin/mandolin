package org.mitre.mandolin.glp.spark

import org.mitre.mandolin.glp.GLPModelSettings


/**
 * @author wellner
 */
object Mandolin extends org.mitre.mandolin.config.LogInit {

  def main(args: Array[String]): Unit = {
    val appSettings = new GLPModelSettings(args)
    val mode = appSettings.appMode
    val numPartitions = appSettings.numPartitions
    val distProcessor = new DistributedProcessor(numPartitions)
      mode match {
        case "train" => distProcessor.processTrain(appSettings)
        case "decode" => distProcessor.processDecode(appSettings)
        case "train-test" => distProcessor.processTrainTest(appSettings)
        case "train-decode" => distProcessor.processTrainDecode(appSettings)
    }
    System.exit(0)
  }
}

package org.mitre.mandolin.glp.local

import org.mitre.mandolin.glp.GLPModelSettings


/**
 * @author wellner
 */
object MandolinWithoutSpark extends org.mitre.mandolin.app.AppMain {

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
      throw new RuntimeException("Mandolin mode set to use Spark, but non-Spark main called")      
    }
    System.exit(0)
  }
}

package org.mitre.mandolin.mx

import ml.dmlc.mxnet.{ DataIter, Context, Shape, IO, FactorScheduler, Model }
import ml.dmlc.mxnet.optimizer._
import com.typesafe.config.Config
import scala.collection.JavaConversions._
import net.ceedubs.ficus.Ficus._

trait MxNetSetup {
  val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  
  def getDeviceArray(appSettings: MxModelSettings) : Array[Context] = {
    val gpuContexts = appSettings.getGpus map {i => Context.gpu(i)}    
    val cpuContexts = appSettings.getCpus map {i => Context.cpu(i)}
    if (gpuContexts.length < 1) { // this checks if the current host is a "gpu" host and assigns to processor "0"
      if (appSettings.gpuHostMapping != null) {
        val curMachineName = java.net.InetAddress.getLocalHost().getHostName
        val config = appSettings.config.as[List[Config]]("mandolin.mx.gpu-host-map")
        var gpuList : List[Int] = Nil
        try {
        config.foreach {c => 
          val g1 = c.getString("gpu")
          if (curMachineName equals g1) {
            val deviceList = c.getIntList("devices")
            gpuList = deviceList.toList map {_.toInt}
          }
        }} catch {case _: Throwable => }
        if (gpuList.length > 0) {
          (gpuList map {e => Context.gpu(e)}).toArray
        } else Context.cpu(0)
        //val specList = config.as[Config]
        //val specList = appSettings.config.as[Config]
        // val gpuHostMaps : List[Config] = appSettings.gpuHostMapping.toList
      } else {
      val ghosts = appSettings.gpuHosts.toSet
      logger.info("GPU hosts: " + ghosts)
      if (ghosts.size > 0) {
        val curMachineName = java.net.InetAddress.getLocalHost().getHostName
        logger.info("Current machine name = " + curMachineName)
        if (ghosts.contains(curMachineName)) Context.gpu(0)
        else cpuContexts.toArray
      } else cpuContexts.toArray
      }
    } else (gpuContexts ++ cpuContexts).toArray         
  }
  
  def getOptimizer(appSettings: MxModelSettings) = {
    val lr = appSettings.mxInitialLearnRate
    val rescale = appSettings.mxRescaleGrad
    appSettings.mxOptimizer match {      
      case "nag" => new NAG(learningRate = lr, momentum = appSettings.mxMomentum, wd = 0.0001f)
      case "adadelta" => new AdaDelta(rho = appSettings.mxRho, rescaleGradient = rescale)
      case "rmsprop" => new RMSProp(learningRate = lr, rescaleGradient = rescale)
      case "adam" => new Adam(learningRate = lr, clipGradient = appSettings.mxGradClip)
      case "adagrad" => new AdaGrad(learningRate = lr, rescaleGradient = rescale)
      case "sgld" => new SGLD(learningRate = lr, rescaleGradient = rescale, clipGradient = appSettings.mxGradClip)
      case _ => new SGD(learningRate = lr, momentum = appSettings.mxMomentum, wd = 0.0001f)
    }
  }
  
  def getTrainIO(appSettings: MxModelSettings, dataShape: Shape) = {
    val preProcessThreads = appSettings.preProcThreads
    val mp = Map(
      "path_imgrec" -> appSettings.trainFile.get,
      "label_name" -> "softmax_label",
      "mean_img" -> appSettings.meanImgFile,
      "data_shape" -> dataShape.toString,
      "batch_size" -> appSettings.miniBatchSize.toString,
      "rand_crop" -> "True",
      "rand_mirror" -> "True",
      "shuffle" -> "True",
      "preprocess_threads" -> preProcessThreads.toString
      )    
    val mp1 = if (appSettings.mxResizeShortest > 0) mp + ("resize" -> appSettings.mxResizeShortest.toString) else mp
    IO.ImageRecordIter(mp1)
  }
  
  def getTestIO(appSettings: MxModelSettings, dataShape: Shape) = {
    val preProcessThreads = appSettings.preProcThreads
    val mp = Map(
      "path_imgrec" -> appSettings.testFile.get,
      "label_name" -> "softmax_label",
      "mean_img" -> appSettings.meanImgFile,
      "data_shape" -> dataShape.toString,
      "batch_size" -> appSettings.miniBatchSize.toString,
      "rand_crop" -> "False",
      "rand_mirror" -> "False",
      "shuffle" -> "False",
      "preprocess_threads" -> preProcessThreads.toString
      )
    val mp1 = if (appSettings.mxResizeShortest > 0) mp + ("resize" -> appSettings.mxResizeShortest.toString) else mp
    IO.ImageRecordIter(mp1)
  }
  
  def getMNISTTrainIO(appSettings: MxModelSettings, dataShape: Shape) = {
    val flat = if (dataShape.size == 3) "False" else "True"
    IO.MNISTIter(Map(
      "image" -> (appSettings.trainFile.get),
      "label" -> (appSettings.mxTrainLabels.get),
      "label_name" -> "softmax_label",
      "input_shape" -> dataShape.toString,
      "batch_size" -> appSettings.miniBatchSize.toString,
      "shuffle" -> "True",
      "flat" -> flat))
  }
  
  def getMNISTTestIO(appSettings: MxModelSettings, dataShape: Shape) = {
    val flat = if (dataShape.size == 3) "False" else "True"
    IO.MNISTIter(Map(
      "image" -> (appSettings.testFile.get),
      "label" -> (appSettings.mxTestLabels.get),
      "label_name" -> "softmax_label",
      "input_shape" -> dataShape.toString,
      "batch_size" -> appSettings.miniBatchSize.toString,
      "shuffle" -> "False",
      "flat" -> flat))
      // "num_parts" -> kv.numWorkers.toString,
      // "part_index" -> kv.`rank`.toString))
  }
   
  
}
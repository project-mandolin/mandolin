package org.mitre.mandolin.mx

import ml.dmlc.mxnet.{ DataIter, Context, Shape, IO, FactorScheduler, Model }
import ml.dmlc.mxnet.optimizer._

trait MxNetSetup {
  def getDeviceArray(appSettings: MxModelSettings) : Array[Context] = {
    val gpuContexts = appSettings.getGpus map {i => Context.gpu(i)}
    val cpuContexts = appSettings.getCpus map {i => Context.cpu(i)}
    (gpuContexts ++ cpuContexts).toArray
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
    IO.ImageRecordIter(Map(
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
    )
  }
  
  def getTestIO(appSettings: MxModelSettings, dataShape: Shape) = {
    val preProcessThreads = appSettings.preProcThreads
    IO.ImageRecordIter(Map(
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
    )
  }
   
  
}
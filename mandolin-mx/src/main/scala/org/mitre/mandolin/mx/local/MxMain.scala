package org.mitre.mandolin.mx.local

import org.mitre.mandolin.mx.{ MxModelSettings, SymbolBuilder, MxNetOptimizer, MxNetWeights, MxNetEvaluator }
import ml.dmlc.mxnet.{ Context, Shape, IO, FactorScheduler, Model }
import ml.dmlc.mxnet.optimizer.SGD

object MxMain {
  
  def getDeviceArray(appSettings: MxModelSettings) : Array[Context] = {
    val gpuContexts = appSettings.gpus map {i => Context.gpu(i)}
    val cpuContexts = appSettings.cpus map {i => Context.cpu(i)}
    (gpuContexts ++ cpuContexts).toArray
  }
  
  def getTrainIO(appSettings: MxModelSettings, dataShape: Shape) = {
    val preProcessThreads = appSettings.preProcThreads.getOrElse(4)
    IO.ImageRecordIter(Map(
      "path_imgrec" -> appSettings.trainFile.get,
      "label_name" -> "softmax_label",
      "mean_img" -> appSettings.meanImgFile.get,
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
    val preProcessThreads = appSettings.preProcThreads.getOrElse(4)
    IO.ImageRecordIter(Map(
      "path_imgrec" -> appSettings.testFile.get,
      "label_name" -> "softmax_label",
      "mean_img" -> appSettings.meanImgFile.get,
      "data_shape" -> dataShape.toString,
      "batch_size" -> appSettings.miniBatchSize.toString,
      "rand_crop" -> "False",
      "rand_mirror" -> "False",
      "shuffle" -> "False",
      "preprocess_threads" -> preProcessThreads.toString
      )
    )
  }
  
  def trainImageModel(appSettings: MxModelSettings) = {
    val devices = getDeviceArray(appSettings)
    val sym     = (new SymbolBuilder).symbolFromSpec(appSettings.config)
    val shape   = Shape(appSettings.channels.get, appSettings.xdim.get, appSettings.ydim.get)
    val trIter = getTrainIO(appSettings, shape)
    val tstIter = getTestIO(appSettings, shape)
    val scheduler = new FactorScheduler(trIter.size, 0.94f) // update by 0.94 after each epoch
    //val scheduler = new MXMultiFactorScheduler(trIter.size, )
    val sgd = new SGD(learningRate = 0.01f, momentum = 0.9f, wd = 0.0001f, clipGradient = 8.0f, lrScheduler = scheduler)
    val updater = new MxNetOptimizer(sgd)
    val weights = new MxNetWeights(1.0f)
    val evaluator = new MxNetEvaluator(sym, devices, shape, appSettings.miniBatchSize, Some("model"))
    evaluator.evaluateTrainingMiniBatch(trIter, weights, updater, appSettings.numEpochs)
    Model.saveCheckpoint(appSettings.modelFile.get, appSettings.numEpochs, sym, weights.getArgParams, weights.getAuxParams)
  }
  
  def main(args: Array[String]) : Unit = {
    val appSettings = new MxModelSettings(args)
    val mode = appSettings.appMode
    mode match {
      case "train" => 
        appSettings.inputType match {
          case Some("recordio") => trainImageModel(appSettings)
          case _ => throw new RuntimeException("Only image models with 'recordio' format currently supported")
        }
      case _ => throw new RuntimeException("Only 'train' mode currently supported")
    }
  }

}
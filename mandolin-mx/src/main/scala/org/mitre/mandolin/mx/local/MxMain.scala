package org.mitre.mandolin.mx.local

import org.mitre.mandolin.glp.{ GLPTrainerBuilder, GLPFactor }
import org.mitre.mandolin.util.LocalIOAssistant
import org.mitre.mandolin.mx.{ GLPFactorIter, MxModelSettings, SymbolBuilder, MxNetOptimizer, MxNetWeights, MxNetEvaluator }
import ml.dmlc.mxnet.{ DataIter, Context, Shape, IO, FactorScheduler, Model }
import ml.dmlc.mxnet.optimizer._

object MxMain extends org.mitre.mandolin.config.LogInit {
  
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
    val evaluator = new MxNetEvaluator(sym, devices, shape, appSettings.miniBatchSize, appSettings.modelFile, appSettings.saveFreq.getOrElse(1))
    evaluator.evaluateTrainingMiniBatch(trIter, tstIter, weights, updater, appSettings.numEpochs)
    Model.saveCheckpoint(appSettings.modelFile.get, appSettings.numEpochs, sym, weights.getArgParams, weights.getAuxParams)
  }
  
  def getVecIOs(appSettings: MxModelSettings) : (Vector[GLPFactor], Vector[GLPFactor], Int) = {
    val io = new LocalIOAssistant
    val components = GLPTrainerBuilder.getComponentsViaSettings(appSettings, io)
    val featureExtractor = components.featureExtractor
    val trFile = appSettings.trainFile.get
    val tstFile = appSettings.testFile.getOrElse(trFile)
    val trVecs = (io.readLines(trFile) map { l => featureExtractor.extractFeatures(l) } toVector)
    val tstVecs = (io.readLines(tstFile) map { l => featureExtractor.extractFeatures(l) } toVector)
    (trVecs, tstVecs, featureExtractor.getNumberOfFeatures)
  }
  
  def getOptimizer(appSettings: MxModelSettings) = {
    val lr = appSettings.mxInitialLearnRate.getOrElse(appSettings.initialLearnRate)
    val rescale = appSettings.mxRescaleGrad.getOrElse(1.0f)
    appSettings.mxOptimizer match {      
      case Some("nag") => new NAG(learningRate = lr, momentum = appSettings.mxMomentum.getOrElse(0.9f), wd = 0.0001f)
      case Some("adadelta") => new AdaDelta(rho = appSettings.mxRho.getOrElse(0.05f), rescaleGradient = rescale)
      case Some("rmsprop") => new RMSProp(learningRate = lr, rescaleGradient = rescale)
      case Some("adam") => new Adam(learningRate = lr, clipGradient = appSettings.mxGradClip.getOrElse(0f))
      case Some("adagrad") => new AdaGrad(learningRate = lr, rescaleGradient = rescale)
      case Some("sgld") => new SGLD(learningRate = lr, rescaleGradient = rescale, clipGradient = appSettings.mxGradClip.getOrElse(0f))
      case _ => new SGD(learningRate = lr, momentum = appSettings.mxMomentum.getOrElse(0.9f), wd = 0.0001f)
    }
  }
  
  def trainGlpModel(appSettings: MxModelSettings) = {
    val devices = getDeviceArray(appSettings)
    val sym     = (new SymbolBuilder).symbolFromSpec(appSettings.config)        
    val (trVecs, tstVecs, nfs) = getVecIOs(appSettings)
    val shape = Shape(nfs)
    val trIter = new GLPFactorIter(trVecs.toIterator, shape, appSettings.miniBatchSize)
    val tstIter = new GLPFactorIter(tstVecs.toIterator, shape, appSettings.miniBatchSize)
    val lr = appSettings.initialLearnRate
    val opt = getOptimizer(appSettings)
    val updater = new MxNetOptimizer(opt)
    val weights = new MxNetWeights(1.0f)
    val evaluator = new MxNetEvaluator(sym, devices, shape, appSettings.miniBatchSize, appSettings.modelFile, appSettings.saveFreq.getOrElse(1))
    evaluator.evaluateTrainingMiniBatch(trIter, tstIter, weights, updater, appSettings.numEpochs)
    Model.saveCheckpoint(appSettings.modelFile.get, appSettings.numEpochs, sym, weights.getArgParams, weights.getAuxParams)
  }
  
  def main(args: Array[String]) : Unit = {
    
    val appSettings = new MxModelSettings(args)
    val mode = appSettings.appMode
    mode match {
      case "train" => 
        appSettings.inputType match {
          case Some("recordio") => trainImageModel(appSettings)
          case _ => trainGlpModel(appSettings)
          // case _ => throw new RuntimeException("Only image models with 'recordio' format currently supported")
        }
      case _ => throw new RuntimeException("Only 'train' mode currently supported")
    }
  }

}
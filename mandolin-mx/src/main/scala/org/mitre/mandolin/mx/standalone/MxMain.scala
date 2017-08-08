package org.mitre.mandolin.mx.standalone

import org.mitre.mandolin.config.LogInit
import org.mitre.mandolin.mlp.{ MMLPTrainerBuilder, MMLPFactor }
import org.mitre.mandolin.util.LocalIOAssistant
import org.mitre.mandolin.mx.{ MxNetSetup, MMLPFactorIter, MxModelSettings, SymbolBuilder, MxNetOptimizer, MxNetWeights, MxNetEvaluator }
import ml.dmlc.mxnet.{ DataIter, Context, Shape, IO, FactorScheduler, Model, Uniform, Xavier }
import ml.dmlc.mxnet.optimizer._

object MxMain extends LogInit with MxNetSetup with org.mitre.mandolin.app.AppMain {  
  
  def trainImageModel(appSettings: MxModelSettings) = {
    val devices = getDeviceArray(appSettings)
    val sym     = (new SymbolBuilder).symbolFromSpec(appSettings.config)
    val shape   = Shape(appSettings.channels, appSettings.xdim, appSettings.ydim)
    val trIter = getTrainIO(appSettings, shape)
    val tstIter = getTestIO(appSettings, shape)    
    val scheduler = new FactorScheduler(trIter.size, 0.94f) // update by 0.94 after each epoch
    //val sgd = new SGD(learningRate = 0.01f, momentum = 0.9f, wd = 0.0001f, clipGradient = 8.0f, lrScheduler = scheduler)
    val opt = getOptimizer(appSettings)
    val updater = new MxNetOptimizer(opt)
    val weights = new MxNetWeights(1.0f)
    val initializer = appSettings.mxInitializer match {
      case "xavier" => new Xavier(rndType = "gaussian", factorType = "in", magnitude = 1.8f)
      case _ => new Uniform(0.01f)
    }
    val evaluator = new MxNetEvaluator(sym, devices, shape, appSettings.miniBatchSize, initializer, appSettings.modelFile, appSettings.saveFreq)
    evaluator.evaluateTrainingMiniBatch(trIter, tstIter, weights, updater, appSettings.numEpochs)
    Model.saveCheckpoint(appSettings.modelFile.get, appSettings.numEpochs, sym, weights.getArgParams, weights.getAuxParams)
  }
  
  def trainMnistModel(appSettings: MxModelSettings) = {
    val devices = getDeviceArray(appSettings)
    val sym     = (new SymbolBuilder).symbolFromSpec(appSettings.config)
    val xd = appSettings.xdim
    val yd = appSettings.ydim
    // if yd = 0 then assume input is flattened and use the xdim as its dimensionality
    // if yd > 0 assume data has shape
    val shape   = if (yd > 0) Shape(appSettings.channels, xd, yd) else Shape(xd) 
    val trIter = getMNISTTrainIO(appSettings, shape)
    val tstIter = getMNISTTestIO(appSettings, shape)    
    //val scheduler = new FactorScheduler(trIter.size, 0.94f) // update by 0.94 after each epoch
    //val sgd = new SGD(learningRate = 0.01f, momentum = 0.9f, wd = 0.0001f, clipGradient = 8.0f, lrScheduler = scheduler)
    val opt = getOptimizer(appSettings)
    val updater = new MxNetOptimizer(opt)
    val weights = new MxNetWeights(1.0f)
    val initializer = appSettings.mxInitializer match {
      case "xavier" => new Xavier(rndType = "gaussian", factorType = "in", magnitude = 1.8f)
      case _ => new Uniform(0.01f)
    }
    val evaluator = new MxNetEvaluator(sym, devices, shape, appSettings.miniBatchSize, initializer, appSettings.modelFile, appSettings.saveFreq)
    evaluator.evaluateTrainingMiniBatch(trIter, tstIter, weights, updater, appSettings.numEpochs)
    Model.saveCheckpoint(appSettings.modelFile.get, appSettings.numEpochs, sym, weights.getArgParams, weights.getAuxParams)
  }
  
  
  def getVecIOs(appSettings: MxModelSettings) : (Vector[MMLPFactor], Vector[MMLPFactor], Int) = {
    val io = new LocalIOAssistant
    val components = MMLPTrainerBuilder.getComponentsViaSettings(appSettings, io)
    val featureExtractor = components.featureExtractor
    val trFile = appSettings.trainFile.get
    val tstFile = appSettings.testFile.getOrElse(trFile)
    val trVecs = (io.readLines(trFile) map { l => featureExtractor.extractFeatures(l) } toVector)
    val tstVecs = (io.readLines(tstFile) map { l => featureExtractor.extractFeatures(l) } toVector)
    (trVecs, tstVecs, featureExtractor.getNumberOfFeatures)
  } 
  
  def trainGlpModel(appSettings: MxModelSettings) = {
    val devices = getDeviceArray(appSettings)
    val sym     = (new SymbolBuilder).symbolFromSpec(appSettings.config)        
    val (trVecs, tstVecs, nfs) = getVecIOs(appSettings)
    val shape = Shape(nfs)
    val trIter = new MMLPFactorIter(trVecs.toIterator, shape, appSettings.miniBatchSize)
    val tstIter = new MMLPFactorIter(tstVecs.toIterator, shape, appSettings.miniBatchSize)
    val lr = appSettings.initialLearnRate
    val opt = getOptimizer(appSettings)
    val updater = new MxNetOptimizer(opt)
    val weights = new MxNetWeights(1.0f)
    val initializer = appSettings.mxInitializer match {
      case "xavier" => new Xavier(rndType = "gaussian", factorType = "in", magnitude = 1.8f)
      case _ => new Uniform(0.01f)
    }
    val evaluator = new MxNetEvaluator(sym, devices, shape, appSettings.miniBatchSize, initializer, appSettings.modelFile, appSettings.saveFreq)
    evaluator.evaluateTrainingMiniBatch(trIter, tstIter, weights, updater, appSettings.numEpochs)
    Model.saveCheckpoint(appSettings.modelFile.get, appSettings.numEpochs, sym, weights.getArgParams, weights.getAuxParams)
  }
  
  def main(args: Array[String]) : Unit = {
    
    val appSettings = new MxModelSettings(args)
    val mode = appSettings.appMode
    mode match {
      case "train" => 
        appSettings.inputType match {
          case "recordio" => trainImageModel(appSettings)
          case "mnist" => trainMnistModel(appSettings)
          case _ => trainGlpModel(appSettings)
          // case _ => throw new RuntimeException("Only image models with 'recordio' format currently supported")
        }
      case _ => throw new RuntimeException("Only 'train' mode currently supported")
    }
  }

}
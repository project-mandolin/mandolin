package org.mitre.mandolin.glp.local
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.util.{ StdAlphabet, RandomAlphabet, Alphabet, IOAssistant, LocalIOAssistant }
import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.predict.DiscreteConfusion
import org.mitre.mandolin.optimize.local.{VectorData}
import org.mitre.mandolin.predict.local.{ LocalTrainer, LocalTrainTester, LocalTrainDecoder, LocalPosteriorDecoder }
import org.mitre.mandolin.glp.{AbstractProcessor, MandolinMLPSettings, GLPModelWriter, CategoricalGLPPredictor, GLPModelReader,
  GLPPosteriorOutputConstructor, GLPFactor, GLPWeights, GLPInstanceEvaluator, GLPLossGradient, GLPModelSpec, ANNetwork, NullGLPUpdater }
import org.mitre.mandolin.optimize.{BatchEvaluator, GenData, Updater}
import com.twitter.chill.EmptyScalaKryoInstantiator
import org.mitre.mandolin.config.MandolinRegistrator
import com.esotericsoftware.kryo.Kryo
import com.twitter.chill.AllScalaRegistrar

class LocalRegistrator extends MandolinRegistrator {
  def registerClasses(kryo: Kryo) = register(kryo)
}

/**
 * @author wellner
 */
class LocalGLPModelWriter extends GLPModelWriter {
  val instantiator = new EmptyScalaKryoInstantiator

  val kryo = {
    val k = instantiator.newKryo()
      k.setClassLoader(Thread.currentThread.getContextClassLoader)
      k
  }
  
  val registrator = new LocalRegistrator
  registrator.registerClasses(kryo)

  def writeModel(weights: GLPWeights): Unit = {
    throw new RuntimeException("Intermediate model writing not implemented with GLPWeights")
  }

  def writeModel(io: IOAssistant, filePath: String, w: GLPWeights, la: Alphabet, ann: ANNetwork, fe: FeatureExtractor[String, GLPFactor]): Unit = {
    io.writeSerializedObject(kryo, filePath, GLPModelSpec(w, ann, la, fe))
  }
}

class LocalGLPModelReader {
  val instantiator = new EmptyScalaKryoInstantiator
  
  val registrator = new LocalRegistrator 
  
  val kryo = {
    val k = instantiator.newKryo()
      k.setClassLoader(Thread.currentThread.getContextClassLoader)
      k
  }

  registrator.registerClasses(kryo)


  def readModel(f: String, io: IOAssistant): GLPModelSpec = {
    io.readSerializedObject(kryo, f, classOf[GLPModelSpec]).asInstanceOf[GLPModelSpec]
  }
}


/**
 * @author wellner
 */
class LocalProcessor extends AbstractProcessor {        
  
  def processTrain(appSettings: MandolinMLPSettings) = {
    if (appSettings.modelFile.isEmpty) throw new RuntimeException("Model file required in training mode")
    if (appSettings.trainFile.isEmpty) throw new RuntimeException("Training file required in training mode")
    val io = new LocalIOAssistant
    val components = getComponentsViaSettings(appSettings, io)       
    val trainFile = appSettings.trainFile
    val fe = components.featureExtractor
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(appSettings, components.ann)
    val lines = io.readLines(trainFile.get).toVector
    val trainer = new LocalTrainer(fe, optimizer)
    val (finalWeights,_) = trainer.trainWeights(lines)
    val modelWriter = new LocalGLPModelWriter
    modelWriter.writeModel(io, appSettings.modelFile.get, finalWeights, components.labelAlphabet, components.ann, fe)
    finalWeights
  }  
  
  def processDecode(appSettings: MandolinMLPSettings) = {
    if (appSettings.modelFile.isEmpty) throw new RuntimeException("Model file required in decoding mode")
    val io = new LocalIOAssistant
    val modelSpec = (new LocalGLPModelReader).readModel(appSettings.modelFile.get, io)
    val testLines = appSettings.testFile map { tf => io.readLines(tf).toVector }
    val predictor = new CategoricalGLPPredictor(modelSpec.ann, true)
    val oc = new GLPPosteriorOutputConstructor()
    val decoder = new LocalPosteriorDecoder(modelSpec.fe, predictor, oc)
    val os = io.getPrintWriterFor(appSettings.outputFile.get, false)
    val outputs = decoder.run(testLines.get, modelSpec.wts)
    writeOutputs(os, outputs.toIterator, Some(modelSpec.la))
    os.close()
  }
  
  def processTrainTest(appSettings: MandolinMLPSettings) = {
    val trainFile = appSettings.trainFile
    if (trainFile.isEmpty) throw new RuntimeException("Training file required in train-test mode")
    val io = new LocalIOAssistant
    val components = getComponentsViaSettings(appSettings, io)
    println("Received components")
    val fe = components.featureExtractor
    val pr = components.predictor
    val optimizer = LocalGLPOptimizer.getLocalOptimizer(appSettings, components.ann)
    val lines = io.readLines(trainFile.get).toVector
    val trainer = new LocalTrainer(fe, optimizer)
    val testLines = appSettings.testFile map { tf => io.readLines(tf).toVector }
    val trainTester =
          new LocalTrainTester[String, GLPFactor, GLPWeights, Int, DiscreteConfusion](trainer,
            pr, appSettings.numEpochs, appSettings.testFreq, appSettings.progressFile)
    trainTester.trainAndTest(lines, testLines.get)
  }
  
  /**
   * A convenience method that trains and evaluates N separate models and test files
   * in a parallel directory structure.
   */
  def processTrainTestDirectories(appSettings: MandolinMLPSettings) = {
    val dir: java.io.File = new java.io.File(appSettings.trainFile.get)
    val io = new LocalIOAssistant
    val testDir: java.io.File = new java.io.File(appSettings.testFile.get)
    val outFile = new java.io.File(appSettings.outputFile.get)
    val trainFiles = dir.listFiles().toVector.map { _.getPath }.sorted
    val testFiles = testDir.listFiles().toVector.map { _.getPath }.sorted
    var labelsSet = false
    val os = new java.io.PrintWriter(outFile)
    (trainFiles zip testFiles) foreach {
        case (tr, tst) =>
          val tstFile = new java.io.File(tst)
          val trLines = scala.io.Source.fromFile(tr).getLines.toVector
          val components = getComponentsViaSettings(appSettings, io)

          val fe = components.featureExtractor
          val pr = components.predictor
          // XXX - this will only work properly if each train-test split has the same label set as the first fold
          val optimizer = LocalGLPOptimizer.getLocalOptimizer(appSettings, components.ann)
          val lines = trLines.toVector
          val trainer = new LocalTrainer(fe, optimizer)
          val testLines = scala.io.Source.fromFile(tst).getLines().toVector
          val trainTesterDecoder =
            new LocalTrainDecoder[String, GLPFactor, GLPWeights, Int](trainer, pr, new GLPPosteriorOutputConstructor, appSettings.numEpochs)
          val outputs = trainTesterDecoder.trainAndDecode(lines, testLines)
          if (!labelsSet) {
            val labelHeader = components.labelAlphabet.getMapping.toSeq.sortWith((a, b) => a._2 < b._2).map(_._1) 
            os.print("ID")
            for (i <- 0 until labelHeader.length) {
              os.print(',')
              os.print(labelHeader(i))
            }
            os.print(',')
            os.print("Label")
            os.println
            labelsSet = true
          }
          outputs foreach { case (s, factor) => os.print(s); os.print(','); os.print(factor.getOneHot.toString); os.println }          
    }
    os.close()
  }
  
  def processTrainDecode(appSettings: MandolinMLPSettings) = {
    val weights = processTrain(appSettings)
    val io = new LocalIOAssistant
    val testLines = appSettings.testFile map { tf => io.readLines(tf).toVector }
    val components = getComponentsViaSettings(appSettings, io)
    val fe = components.featureExtractor
    val predictor = components.predictor
    val oc = components.outputConstructor
    val decoder = new LocalPosteriorDecoder(fe, predictor, oc)
    val os = io.getPrintWriterFor(appSettings.outputFile.get, false)
    val outputs = decoder.run(testLines.get, weights)
    writeOutputs(os, outputs.toIterator, Some(components.labelAlphabet))
    os.close()
  }
}

class GlpLocalBatchEvaluator(ev: GLPInstanceEvaluator[NullGLPUpdater]) 
extends BatchEvaluator[GLPFactor, GLPWeights, GLPLossGradient] {
  val up = new NullGLPUpdater
  def evaluate(gd:GenData[GLPFactor], w: GLPWeights) : GLPLossGradient = {
    gd match {
      case data: VectorData[GLPFactor] =>
        data.vec map {d => ev.evaluateTrainingUnit(d, w, up)} reduce {_ ++ _}
      case _ => throw new RuntimeException("Require local Scala vector data sequence for LocalBatchEvaluator")
    }
  }
}



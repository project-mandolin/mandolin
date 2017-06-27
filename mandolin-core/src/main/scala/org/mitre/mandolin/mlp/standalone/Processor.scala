package org.mitre.mandolin.mlp.standalone

/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.util.{Alphabet, IOAssistant, LocalIOAssistant}
import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.predict.DiscreteConfusion
import org.mitre.mandolin.optimize.standalone.VectorData
import org.mitre.mandolin.predict.standalone.{Trainer, TrainTester, TrainDecoder, PosteriorDecoder}
import org.mitre.mandolin.mlp.{AbstractProcessor, MandolinMLPSettings, MMLPModelWriter, CategoricalMMLPPredictor, MMLPPosteriorOutputConstructor, MMLPFactor, MMLPWeights, MMLPInstanceEvaluator, MMLPLossGradient, MMLPModelSpec, ANNetwork, NullMMLPUpdater}
import org.mitre.mandolin.optimize.{BatchEvaluator, GenData}
import com.twitter.chill.EmptyScalaKryoInstantiator
import org.mitre.mandolin.config.MandolinRegistrator
import com.esotericsoftware.kryo.Kryo

class Registrator extends MandolinRegistrator {
  def registerClasses(kryo: Kryo) = register(kryo)
}

/**
  * @author wellner
  */
class StandaloneMMLPModelWriter extends MMLPModelWriter {
  val instantiator = new EmptyScalaKryoInstantiator

  val kryo = {
    val k = instantiator.newKryo()
    k.setClassLoader(Thread.currentThread.getContextClassLoader)
    k
  }

  val registrator = new Registrator
  registrator.registerClasses(kryo)

  def writeModel(weights: MMLPWeights): Unit = {
    throw new RuntimeException("Intermediate model writing not implemented with MMLPWeights")
  }

  def writeModel(io: IOAssistant, filePath: String, w: MMLPWeights, la: Alphabet, ann: ANNetwork, fe: FeatureExtractor[String, MMLPFactor]): Unit = {
    io.writeSerializedObject(kryo, filePath, MMLPModelSpec(w, ann, la, fe))
  }
}

class StandaloneMMLPModelReader {
  val instantiator = new EmptyScalaKryoInstantiator

  val registrator = new Registrator

  val kryo = {
    val k = instantiator.newKryo()
    k.setClassLoader(Thread.currentThread.getContextClassLoader)
    k
  }

  registrator.registerClasses(kryo)


  def readModel(f: String, io: IOAssistant): MMLPModelSpec = {
    io.readSerializedObject(kryo, f, classOf[MMLPModelSpec]).asInstanceOf[MMLPModelSpec]
  }
}


/**
  * @author wellner
  */
class Processor extends AbstractProcessor {

  def processTrain(appSettings: MandolinMLPSettings) = {
    if (appSettings.modelFile.isEmpty) throw new RuntimeException("Model file required in training mode")
    if (appSettings.trainFile.isEmpty) throw new RuntimeException("Training file required in training mode")
    val io = new LocalIOAssistant
    val components = getComponentsViaSettings(appSettings, io)
    val trainFile = appSettings.trainFile
    val fe = components.featureExtractor
    val optimizer = MMLPOptimizer.getOptimizer(appSettings, components.ann)
    val lines = io.readLines(trainFile.get).toVector
    val trainer = new Trainer(fe, optimizer)
    val (finalWeights, _) = trainer.trainWeights(lines)
    val modelWriter = new StandaloneMMLPModelWriter
    modelWriter.writeModel(io, appSettings.modelFile.get, finalWeights, components.labelAlphabet, components.ann, fe)
    finalWeights
  }

  def processDecode(appSettings: MandolinMLPSettings) = {
    if (appSettings.modelFile.isEmpty) throw new RuntimeException("Model file required in decoding mode")
    val io = new LocalIOAssistant
    val modelSpec = (new StandaloneMMLPModelReader).readModel(appSettings.modelFile.get, io)
    val testLines = appSettings.testFile map { tf => io.readLines(tf).toVector }
    val predictor = new CategoricalMMLPPredictor(modelSpec.ann, true)
    val oc = new MMLPPosteriorOutputConstructor()
    val decoder = new PosteriorDecoder(modelSpec.fe, predictor, oc)
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
    val optimizer = MMLPOptimizer.getOptimizer(appSettings, components.ann)
    val lines = io.readLines(trainFile.get).toVector
    val trainer = new Trainer(fe, optimizer)
    val testLines = appSettings.testFile map { tf => io.readLines(tf).toVector }
    val trainTester =
      new TrainTester[String, MMLPFactor, MMLPWeights, Int, DiscreteConfusion](trainer,
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
    val trainFiles = dir.listFiles().toVector.map {
      _.getPath
    }.sorted
    val testFiles = testDir.listFiles().toVector.map {
      _.getPath
    }.sorted
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
        val optimizer = MMLPOptimizer.getOptimizer(appSettings, components.ann)
        val lines = trLines.toVector
        val trainer = new Trainer(fe, optimizer)
        val testLines = scala.io.Source.fromFile(tst).getLines().toVector
        val trainTesterDecoder =
          new TrainDecoder[String, MMLPFactor, MMLPWeights, Int](trainer, pr, new MMLPPosteriorOutputConstructor, appSettings.numEpochs)
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
    val decoder = new PosteriorDecoder(fe, predictor, oc)
    val os = io.getPrintWriterFor(appSettings.outputFile.get, false)
    val outputs = decoder.run(testLines.get, weights)
    writeOutputs(os, outputs.toIterator, Some(components.labelAlphabet))
    os.close()
  }
}

class MMLPBatchEvaluator(ev: MMLPInstanceEvaluator[NullMMLPUpdater])
  extends BatchEvaluator[MMLPFactor, MMLPWeights, MMLPLossGradient] {
  val up = new NullMMLPUpdater

  def evaluate(gd: GenData[MMLPFactor], w: MMLPWeights): MMLPLossGradient = {
    gd match {
      case data: VectorData[MMLPFactor] =>
        data.vec map { d => ev.evaluateTrainingUnit(d, w, up) } reduce {
          _ ++ _
        }
      case _ => throw new RuntimeException("Require standalone Scala vector data sequence for LocalBatchEvaluator")
    }
  }
}



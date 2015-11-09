package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.config.AppConfig
import org.mitre.mandolin.util.{ StdAlphabet, PrescaledAlphabet, RandomAlphabet, Alphabet, IdentityAlphabet, AlphabetWithUnitScaling, IOAssistant }
import org.mitre.mandolin.optimize.spark.{
  DistributedOnlineOptimizer,
  DistributedOptimizerEstimator
}
import org.mitre.mandolin.predict.{
  DiscreteConfusion,
  EvalPredictor,
  OutputConstructor,
  Predictor
}

import org.mitre.mandolin.predict.spark.{
  EvalDecoder,
  PosteriorDecoder,
  TrainDecoder,
  TrainTester,  
  Decoder,
  Trainer
}

import org.mitre.mandolin.optimize.local.LocalOnlineOptimizer
import org.mitre.mandolin.predict.local.{ LocalTrainer, LocalTrainTester, LocalTrainDecoder, LocalEvalDecoder, LocalPosteriorDecoder, LocalDecoder }

import org.mitre.mandolin.transform.{ FeatureExtractor, FeatureImportance }
import org.mitre.mandolin.gm.{ Feature, NonUnitFeature }
import org.mitre.mandolin.util.LineParser
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import scala.reflect.ClassTag

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{ FileSystem, Path }

case class GLPComponentSet(
    evaluator: GLPInstanceEvaluator, 
    predictor: GLPPredictor, 
    outputConstructor: GLPPosteriorOutputConstructor, 
    featureExtractor: FeatureExtractor[String,GLPFactor],
    labelAlphabet: Alphabet,
    dim: Int,
    npts: Int)

/**
 * General driver/processor class for GLP and linear models. Includes various utility methods
 * used by subclasses for reading in training/test data, determining GLP model topology from
 * a specification provided in the configuration file and setting/interpreting all of the
 * various configuration options.
 * @param appSettings GLPModelSettings that specify everything for training/decoding/etc.
 * @author wellner
 */
abstract class Processor extends LineParser {

  def getLabelAlphabet(labFile: Option[String] = None, io: IOAssistant): Alphabet = {
    if (labFile.isDefined) {
      val labels = io.readLines(labFile.get)
      val la = new StdAlphabet
      labels foreach { l => la.ofString(l) }
      la.ensureFixed
      la
    } else new StdAlphabet
  }

  def getGLPSpec(cs: List[Map[String, String]], idim: Int, odim: Int): IndexedSeq[LType] = {
    for (i <- 0 until cs.length) yield {
      val l = cs(i)
      val ltype = l("ltype")
      val dim =
        if (i == 0) idim else if (i == (cs.length - 1)) odim
        else {
          if (l.contains("dim"))
            l("dim").toInt
          else throw new RuntimeException("Expected 'dim' provided in layer specification for non-input layer " + i)
        }
      val dropOut = l.get("dropout-ratio") match { case Some(v) => v.toDouble case None => 0.0 }
      val l1Pen = l.get("l1-pen") match { case Some(v) => v.toDouble case None => 0.0 }
      val l2Pen = l.get("l2-pen") match { case Some(v) => v.toDouble case None => 0.0 }
      val mn = l.get("max-norm") match { case Some(v) => v.toDouble case None => 0.0 }
      val cval = l.get("margin-size") match { case Some(v) => v.toDouble case None => 1.0 }
      val rlen = l.get("ramp-width") match { case Some(v) => v.toDouble case None => 2.0 }
      ltype match {
        case "Input"        => LType(InputLType, dim, dropOut)
        case "InputSparse"  => LType(SparseInputLType, dim, dropOut)
        case "TanH"         => LType(TanHLType, dim, dropOut, l1Pen, l2Pen, mn)
        case "Logistic"     => LType(LogisticLType, dim, dropOut, l1Pen, l2Pen, mn)
        case "Linear"       => LType(LinearLType, dim, dropOut, l1Pen, l2Pen, mn)
        case "CrossEntropy" => LType(CrossEntropyLType, dim, dropOut, l1Pen, l2Pen, mn)
        case "Relu"         => LType(ReluLType, dim, dropOut, l1Pen, l2Pen, mn)
        case "SoftMax"      => LType(SoftMaxLType, dim, 0.0, l1Pen, l2Pen, mn)
        case "Hinge"        => LType(HingeLType(cval), dim, 0.0, l1Pen, l2Pen, mn)
        case "ModHuber"     => LType(ModHuberLType, dim, 0.0, l1Pen, l2Pen, mn)
        case "Ramp"         => LType(RampLType(rlen), dim, 0.0, l1Pen, l2Pen, mn)
        case "TransLog"     => LType(TransLogLType, dim, 0.0, l1Pen, l2Pen, mn)
        case "TLogistic"    => LType(TLogisticLType, dim, 0.0, l1Pen, l2Pen, mn)
        case a              => throw new RuntimeException("Unrecognized layer type: " + a)
      }
    }
  }

  def getAlphabet(inputLines: Vector[String], la: StdAlphabet, scaling: Boolean, selectFeatures: Int, fd: Option[String], io: IOAssistant): (Alphabet, Int) = {
    getAlphabet(inputLines.iterator, la, scaling, selectFeatures, fd, io)
  }

  def getAlphabet(inputLines: Iterator[String], la: Alphabet, scaling: Boolean, selectFeatures: Int, fdetailFile: Option[String], io: IOAssistant): (Alphabet, Int) = {
    val alphabet = if (scaling) new AlphabetWithUnitScaling else new StdAlphabet
    var numPoints = 0
    val fbuf = new collection.mutable.ArrayBuffer[Array[Feature]]
    val yvs = new collection.mutable.ArrayBuffer[Int]
    inputLines foreach { l =>
      numPoints += 1
      if (selectFeatures > 0) { // selecting features, so need to instantiate at this point...
        val (y, fv, _) = sparseOfLine(l, alphabet, addBias = false, buildVecs = true) // filter out features with a prefix
        fbuf append fv
        val yi = la.ofString(y)
        yvs append yi
      } else {
        val (y, _, _) = sparseOfLine(l, alphabet, addBias = false, buildVecs = false) // filter out features with a prefix
        la.ofString(y)
      }
    }
    val finalAlphabet =
      if ((selectFeatures > 0) && (selectFeatures < alphabet.getSize)) {
        val featureImpCompute = new FeatureImportance // basic feature selection
        val dataArr = fbuf.toArray
        val yvArr = yvs.toArray
        val numFeatures = alphabet.getSize
        val toKeep = featureImpCompute.computeMutualInfo(dataArr, yvArr, numFeatures, la.getSize, selectFeatures)
        val revMap = alphabet.mapping.foldLeft(Map[Int, String]()) { case (ac, (s, i)) => ac.updated(i, s) }         
        val newAlpha = new StdAlphabet
        toKeep foreach { i => newAlpha.ofString(revMap(i)) } // update newAlphabet only with features to keep
        val finalAlphabet = 
          if (scaling) {
            // if we're scaling features, we need to get the min and max values from original 
            // alphabet and map those into the reduced alphabet with selected features
            val origAlpha = alphabet.asInstanceOf[AlphabetWithUnitScaling]
            val maxVals = Array.fill(toKeep.length)(0.0)
            val minVals = Array.fill(toKeep.length)(0.0)
            toKeep foreach {i =>              
              val ni = newAlpha.ofString(revMap(i))
              maxVals(ni) = origAlpha.fmax(i)
              minVals(ni) = origAlpha.fmin(i)
              }            
            val psAlpha = new PrescaledAlphabet(minVals, maxVals)
            toKeep foreach {i => psAlpha.ofString(revMap(i))}
            psAlpha
          } else newAlpha
        finalAlphabet
      } else alphabet
    fdetailFile foreach { f =>
      val wr = io.getPrintWriterFor(f, false)
      finalAlphabet.mapping foreach {
        case (s, i) =>
          wr.write(s)
          wr.write('\n')
      }
      wr.close()
    }
    println("Feature symbol table extracted ... " + finalAlphabet.getSize + " features identified")
    if (selectFeatures > 0) {
      println((alphabet.getSize - finalAlphabet.getSize).toString + " features removed based on mutual information selection..\n")
    }
    (finalAlphabet,numPoints)
  }

  def getAlphabet(appSettings: GLPModelSettings, la: Alphabet, io: IOAssistant): (Alphabet, Int) = {
    if (appSettings.useRandom) (new RandomAlphabet(appSettings.numFeatures), 1000)
    else
      getAlphabet(io.readLines(appSettings.trainFile.get), la, appSettings.scaleInputs,
        appSettings.filterFeaturesMI, appSettings.printFeatureFile, io)
  }

  def getSubComponents(confSpecs: List[Map[String, String]], idim: Int, odim: Int): (GLPInstanceEvaluator, GLPPredictor, GLPPosteriorOutputConstructor) = {
    val specs = getGLPSpec(confSpecs, idim, odim)
    getSubComponents(specs)
  }

  def getSubComponents(specs: IndexedSeq[LType]): (GLPInstanceEvaluator, GLPPredictor, GLPPosteriorOutputConstructor) = {
    val nn = ANNetwork(specs)
    val evaluator = new GLPInstanceEvaluator(nn)
    val predictor = new GLPPredictor(nn, true)
    val oc = new GLPPosteriorOutputConstructor
    (evaluator, predictor, oc)
  }

  def getComponentsDenseVecs(confSpecs: List[Map[String, String]], dim: Int, labelAlphabet: Alphabet): GLPComponentSet = {
    val (evaluator, predictor, outConstructor) = getSubComponents(confSpecs, dim, labelAlphabet.getSize)
      val fe = new StdVectorExtractorWithAlphabet(labelAlphabet, dim)
      GLPComponentSet(evaluator, predictor, outConstructor, fe, labelAlphabet, dim, 1000)
  }

  def getComponentsDenseVecs(appSettings: GLPModelSettings, io: IOAssistant): GLPComponentSet = {
    val la = getLabelAlphabet(appSettings.labelFile, io)
    getComponentsDenseVecs(appSettings.netspec, appSettings.denseVectorSize, la)
  }

  def getComponentsDenseVecs(layerSpecs: IndexedSeq[LType]): GLPComponentSet = {
    val (evaluator, predictor, outConstructor) = getSubComponents(layerSpecs)
    val labelAlphabet = new IdentityAlphabet(layerSpecs.last.dim)
    val fe = new StdVectorExtractorWithAlphabet(labelAlphabet, layerSpecs.head.dim)
    GLPComponentSet(evaluator, predictor, outConstructor, fe, labelAlphabet, layerSpecs.head.dim,1000)
  }

  def getComponentsHashsedFeatures(confSpecs: List[Map[String, String]], randFeatures: Int, labelAlphabet: StdAlphabet): GLPComponentSet = {
    val fa = new RandomAlphabet(randFeatures)
    val (evaluator, predictor, outConstructor) = getSubComponents(confSpecs, randFeatures, labelAlphabet.getSize)
    val fe = new SparseVecFeatureExtractor(fa, labelAlphabet)
    GLPComponentSet(evaluator, predictor, outConstructor, fe, labelAlphabet, randFeatures, 1000)
  }

  def getComponentsHashedFeatures(appSettings: GLPModelSettings, io: IOAssistant): GLPComponentSet = {
    val la = getLabelAlphabet(appSettings.labelFile, io)
    val fa = new RandomAlphabet(appSettings.numFeatures)
    val (evaluator, predictor, outConstructor) = getSubComponents(appSettings.netspec, appSettings.numFeatures, la.getSize)
    val fe = new SparseVecFeatureExtractor(fa, la)
    GLPComponentSet(evaluator, predictor, outConstructor, fe, la, appSettings.numFeatures,1000)
  }

  def getComponentsInducedAlphabet(confSpecs: List[Map[String, String]], lines: Iterator[String],
                                   la: Alphabet, scale: Boolean = false, selectedFeatures: Int = -1, io: IOAssistant): GLPComponentSet = {
    val (fa,npts) = getAlphabet(lines, la, scale, selectedFeatures, None, io)
    fa.ensureFixed
    la.ensureFixed
    val (evaluator, predictor, outConstructor) = getSubComponents(confSpecs, fa.getSize, la.getSize)
    val isSparse = confSpecs.head("ltype").equals("InputSparse")
    val fe = if (isSparse) new SparseVecFeatureExtractor(fa, la) else new VecFeatureExtractor(fa, la)
    GLPComponentSet(evaluator, predictor, outConstructor, fe, la, fa.getSize, npts)
  }

  def getComponentsInducedAlphabet(appSettings: GLPModelSettings, io: IOAssistant): GLPComponentSet = {
    val la = getLabelAlphabet(appSettings.labelFile, io)
    val lines = io.readLines(appSettings.trainFile.get)
    getComponentsInducedAlphabet(appSettings.netspec, lines, la, appSettings.scaleInputs, appSettings.filterFeaturesMI, io)
  }

  def getComponentsViaSettings(appSettings: GLPModelSettings, io: IOAssistant): GLPComponentSet = {
    if (appSettings.denseVectorSize > 0) getComponentsDenseVecs(appSettings, io)
    else if (appSettings.useRandom) getComponentsHashedFeatures(appSettings, io)
    else getComponentsInducedAlphabet(appSettings, io)
  }

  def writeOutputs(os: IOAssistant#PrintWriter, outputs: Iterator[(String, GLPFactor)], laOpt: Option[Alphabet]): Unit = {
    laOpt foreach { la =>
      val labelHeader = la.getMapping.toSeq.sortWith((a, b) => a._2 < b._2).map(_._1) // get label strings sorted by their index
      os.print("ID")
      for (i <- 0 until labelHeader.length) {
        os.print(',')
        os.print(labelHeader(i))
      }
      os.print(',')
      os.print("Label")
      os.println
    }
    outputs foreach { case (s, factor) => os.print(s); os.print(','); os.print(factor.getOneHot.toString); os.println }
  }
}

/**
 * @author wellner
 */
class DistributedProcessor(val numPartitions: Int) extends Processor {
  def this() = this(0)

  //def processTrain(ev: GLPInstanceEvaluator)

  def processTrain(appSettings: GLPModelSettings) = {
    if (appSettings.trainFile.isEmpty) throw new RuntimeException("Training file required")
    val sc = AppConfig.getSparkContext(appSettings)
    val io = new IOAssistant(Some(sc))
    val components = getComponentsViaSettings(appSettings, io)
    val ev = components.evaluator
    val fe = components.featureExtractor
    val trainFile = appSettings.trainFile.get
    val network = ev.glp
    val optimizer: DistributedOptimizerEstimator[GLPFactor, GLPWeights] = GLPOptimizer.getDistributedOptimizer(sc, appSettings, network, ev)
    val lines =
      if (numPartitions > 0) sc.textFile(trainFile, numPartitions).coalesce(numPartitions, true)
      else sc.textFile(trainFile)
    val trainer = new Trainer[String, GLPFactor, GLPWeights](fe, optimizer, appSettings.storageLevel)
    val (w, _) = trainer.trainWeights(lines)
    val modelWriter = new GLPModelWriter(Some(sc))
    modelWriter.writeModel(io, appSettings.modelFile.get, w, components.labelAlphabet, ev, fe)
    w
  }

  def processDecode(appSettings: GLPModelSettings) = {
    if (appSettings.modelFile.isEmpty) throw new RuntimeException("Model file required as input in decoding mode")
    val sc = AppConfig.getSparkContext(appSettings)
    val io = new IOAssistant(Some(sc))
    val modelSpec = (new GLPModelReader(Some(sc))).readModel(appSettings.modelFile.get, io)
    val testLines = sc.textFile(appSettings.testFile.get, numPartitions).coalesce(numPartitions, true)
    val evaluator = modelSpec.evaluator
    val predictor = new GLPPredictor(evaluator.glp, true)
    val oc = new GLPPosteriorOutputConstructor()
    val decoder = new PosteriorDecoder(modelSpec.fe, predictor, oc)
    val os = io.getPrintWriterFor(appSettings.outputFile.get, false)
    val outputs = decoder.run(testLines, sc.broadcast(modelSpec.wts))
    writeOutputs(os, outputs.toLocalIterator, Some(modelSpec.la))
    os.close()
  }

  def processTrainTest(appSettings: GLPModelSettings) = {
    if (appSettings.trainFile.isEmpty) throw new RuntimeException("Training file required")    
    val sc = AppConfig.getSparkContext(appSettings)
    val io = new IOAssistant(Some(sc))
    val components = getComponentsViaSettings(appSettings, io)
    val ev = components.evaluator
    val fe = components.featureExtractor
    val pr = components.predictor

    val trainFile = appSettings.trainFile
    val optimizer = GLPOptimizer.getDistributedOptimizer(sc, appSettings, ev.glp, ev)
    val lines = sc.textFile(appSettings.trainFile.get, numPartitions).coalesce(numPartitions, true)
    val trainer = new Trainer(fe, optimizer)
    val testLines = sc.textFile(appSettings.testFile.get, numPartitions).coalesce(numPartitions, true)
    val trainTester =
      new TrainTester[String, GLPFactor, GLPWeights, GLPLossGradient, Int, DiscreteConfusion](trainer,
        pr, appSettings.numEpochs, appSettings.testFreq, sc, appSettings.progressFile)
    trainTester.trainAndTest(lines, testLines)
  }

  def processTrainDecode(appSettings: GLPModelSettings) = {
    val sc = AppConfig.getSparkContext(appSettings)
    val io = new IOAssistant(Some(sc))
    val components = getComponentsViaSettings(appSettings, io)
    val ev = components.evaluator
    val fe = components.featureExtractor
    val predictor = components.predictor
    val labelAlphabet = components.labelAlphabet
    val oc = components.outputConstructor
    val weights = processTrain(appSettings)
    val testLines = sc.textFile(appSettings.testFile.get, numPartitions).coalesce(numPartitions, true)
    val decoder = new PosteriorDecoder(fe, predictor, oc)
    val os = io.getPrintWriterFor(appSettings.outputFile.get, false)
    val outputs = decoder.run(testLines, sc.broadcast(weights))
    writeOutputs(os, outputs.toLocalIterator, Some(labelAlphabet))
    os.close()
  }
}

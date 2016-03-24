package org.mitre.mandolin.lcrf.local

import org.mitre.mandolin.glp.AbstractProcessor
import org.mitre.mandolin.config.{LearnerSettings, AppSettings}
import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.util.LocalIOAssistant
import org.mitre.mandolin.optimize.local.LocalOnlineOptimizer
import org.mitre.mandolin.predict.local.{LocalTrainer, LocalDecoder}
import org.mitre.mandolin.predict.{OutputConstructor, Predictor}

import org.mitre.mandolin.lcrf.{CrfWeights, LCrfLossGradient, 
  CrfEvalPredictor,CrfSequenceEvaluator, BasicLCrfSgdUpdater, CrfSeqUtils, CrfSettings,CrfModelWriter, CrfModelReader}

import org.mitre.jcarafe.crf.{MemoryInstanceSequence, 
  TrainingSeqGen, DecodingSeqGen, AbstractInstance, StaticFeatureManagerBuilder,TrainingFactoredFeatureRep,
  CoreModel, StdModel, RandomStdModel, RandomLongAlphabet, JsonSeqGen, TextSeqGen, BloomLexicon, WordProperties, StatelessViterbi}
import org.mitre.jcarafe.util.{ Options, AbstractLabel }


class LocalCrfProcessor extends AbstractProcessor with CrfSeqUtils {
  
  
  def getLabelMap(appSettings: CrfModelSettings, frep: TrainingFactoredFeatureRep[String], crfOpts: Options) = {
    val labelSgen = getSeqGen(appSettings.inFileMode, frep, crfOpts) // get another seqgen just to identify the label alphabet...
    val docStrings = scala.io.Source.fromFile(new java.io.File(appSettings.trainFile.get)).getLines

    var allLabels: Map[AbstractLabel, Int] = Map()
    
    docStrings foreach { str =>
        val sseq = labelSgen.toSources(labelSgen.deserializeFromString(str))
        val la = labelSgen.getLAlphabet.mp
        la foreach { case (k, v) => allLabels += (k -> v) }
    }
    allLabels
  }

  def getMgr(appSettings: CrfModelSettings) = {
    val ss =
      if (appSettings.useRandom)
        Seq("--random-features", "--num-random-features", appSettings.numFeatures.toString,
        "--tagset", appSettings.tagsetFile, "--gaussian-prior", "1E300", "--no-begin")
      else Seq("--tagset", appSettings.tagsetFile, "--gaussian-prior", "1E300", "--no-begin")

    val ss1 = if (appSettings.useRandom) ss ++ (Seq("--no-cache")) else ss

    val crfOptArray = 
      if (appSettings.noPreProc) (ss1 ++ Seq("--no-pre-proc", "--seq-boundary", "s", "--input-file", appSettings.trainFile.get)) 
      else ss1
    val crfOpts = new Options(crfOptArray.toArray)
    val lexicon = appSettings.lexiconDir map { d => new BloomLexicon(d) }
    val wdProps = appSettings.wordPropFile map { d => WordProperties(d) }
    val builder = new StaticFeatureManagerBuilder[String](appSettings.featureSet, lexicon, wdProps, None, None, false)
    (builder.getFeatureManager, crfOpts)    
  }
  
  def getDecoder(appSettings: CrfModelSettings) = {
    val (_, crfOpts) = getMgr(appSettings)
    val isg = getDecodingSeqGen(appSettings.inFileMode, crfOpts)
    isg
  }
  
  def getFeatureExtractor(appSettings: CrfModelSettings) = {
    val (initMgr, crfOpts) = getMgr(appSettings)
    val isg = getSeqGen(appSettings.inFileMode, new TrainingFactoredFeatureRep[String](initMgr, crfOpts), crfOpts)
    val sg =
      if (appSettings.useRandom) {
        val labMgr = new StaticFeatureManagerBuilder[String]("default", None, None, None, None, false).getFeatureManager
        val labMap = getLabelMap(appSettings, new TrainingFactoredFeatureRep[String](labMgr, crfOpts), crfOpts)
        val il = isg.getLAlphabet
        labMap foreach { case (l, i) => il.update(l, i) }
        il.fixed_=(true) // fix alphabet here
        isg
      } else {
        getTables(appSettings, isg, crfOpts)
      }
    new CrfFeatureExtractor(sg)
  }
  
   /*
   * Get label and feature Alphabets from input files. Does not use Spark; all work done on the driver.
   */
  def getTables(appSettings: CrfModelSettings, sg: TrainingSeqGen[String], crfOpts: Options): TrainingSeqGen[String] = {
    val docStrings = scala.io.Source.fromFile(new java.io.File(appSettings.trainFile.get))("UTF-8")
    docStrings.getLines() foreach {str =>
       val sseq = sg.toSources(sg.deserializeFromString(str))
       sseq foreach { ss => sg.processSupportingFeatures(ss) }
    }
    sg
  }

  
  def processTrain(appSettings: CrfModelSettings) = {
    if (appSettings.modelFile.isEmpty) throw new RuntimeException("Model file required in training mode")
    if (appSettings.trainFile.isEmpty) throw new RuntimeException("Training file required in training mode")
    val nfs = appSettings.numFeatures
    val io = new LocalIOAssistant
    val fe = getFeatureExtractor(appSettings)
    val nls = fe.sgen.getNumberOfStates
    val weights = new CrfWeights(Array.fill(nfs)(0.0))
    val evaluator = new CrfSequenceEvaluator(nls, nfs)
    val updater = new BasicLCrfSgdUpdater
    val optimizer = new LocalOnlineOptimizer[MemoryInstanceSequence, CrfWeights, LCrfLossGradient, BasicLCrfSgdUpdater](
        weights, evaluator, updater,10,1,16,None)
    val lines = io.readLines(appSettings.trainFile.get).toVector
    val trainer = new LocalTrainer(fe, optimizer)
    val (finalWeights,_) = trainer.trainWeights(lines)
    val modelWriter = new CrfModelWriter(fe.sgen, nfs)
    val mfile = new java.io.File(appSettings.modelFile.get)
    modelWriter.writeModel(mfile,finalWeights)
    finalWeights
  }
  
  def processDecode(appSettings: CrfModelSettings) = {
    if (appSettings.modelFile.isEmpty) throw new RuntimeException("Model file required in decoding mode")
    if (appSettings.testFile.isEmpty) throw new RuntimeException("Test file required in decoding mode")
    val nfs = appSettings.numFeatures
    val io = new LocalIOAssistant
    val weights = (new CrfModelReader).readModel(new java.io.File(appSettings.modelFile.get))
    val dsg = getDecoder(appSettings)
    val predictor = new CrfPredictor(dsg)
    val decoder = new LocalDecoder(new IdentityFeatureExtractor(nfs), predictor,new IdentityOutputConstructor) 
    val testLines = appSettings.testFile map { tf => io.readLines(tf).toVector }
    val outputs = decoder.run(testLines.get, weights)
    val os = io.getPrintWriterFor(appSettings.outputFile.get, false)
    outputs foreach {s => os.write(s); os.write('\n')}
    os.close
  }
  
  def processTrainTest(appSettings: CrfSettings) = {}

}

class CrfPredictor(sg: DecodingSeqGen[String])  extends Predictor[String,CrfWeights,String] {
  def getPrediction(unit: String, weights: CrfWeights): String = {
    val dobj = sg.deserializeFromRawString(unit)
    val seqs = sg.createSeqsWithInput(dobj)
    val viterbiInstance = new StatelessViterbi(false, 1, sg.getNumberOfFeatures)
    seqs foreach {s => viterbiInstance.assignBestSequence(s.iseq, weights.wts)}
    sg.seqsToString(dobj, seqs)
  }
  def getScoredPredictions(unit: String, weights: CrfWeights) : Seq[(Float,String)] = throw new RuntimeException("Scoring not available on CRF")
}

class IdentityOutputConstructor extends OutputConstructor[String,String,String] {
  def constructOutput(input: String, response: String, tunitStr: String) : String = response
  def responseString(r: String) : String = r
  def intToResponseString(i: Int) : String = throw new RuntimeException("Int response not available with sequences")
}

class IdentityFeatureExtractor(nfs: Int) extends FeatureExtractor[String,String] with Serializable {
  def extractFeatures(s: String) = s
  def getNumberOfFeatures = nfs
  def getAlphabet = throw new RuntimeException("Alphabet not available")
}

class CrfFeatureExtractor(val sgen: TrainingSeqGen[String]) extends FeatureExtractor[String, MemoryInstanceSequence] with Serializable {

  def extractFeatures(s: String) = {
    val ss = sgen.createSeqsWithInput(sgen.deserializeFromString(s))
    val buf = new collection.mutable.ArrayBuffer[AbstractInstance]
    ss.seq foreach { s =>
      val sseq = s.iseq
      if (sseq.length > 0) buf ++= sseq }
    new MemoryInstanceSequence(buf.toIndexedSeq)
  }

  def getNumberOfFeatures = {
    sgen.getNumberOfFeatures
  }
  
  def getAlphabet = throw new RuntimeException("Alphabet not available")
}

class CrfModelSettings(args: Array[String]) extends LearnerSettings(args) with CrfSettings

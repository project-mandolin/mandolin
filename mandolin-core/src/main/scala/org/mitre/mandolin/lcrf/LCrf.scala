package org.mitre.mandolin.lcrf

import org.mitre.mandolin.optimize.{TrainingUnitEvaluator, LossGradient, Updater, Weights}
import org.mitre.mandolin.config.AppSettings
import org.mitre.mandolin.util.{SparseTensor1, DenseTensor1}
import org.mitre.mandolin.predict.{ EvalPredictor, DiscreteConfusion }

import org.mitre.jcarafe.crf.{
  AbstractInstance,
  BloomLexicon,
  Viterbi,
  CoreModel,
  MemoryInstanceSequence,
  SparseStatelessCrf,
  StandardSerializer,
  StdModel,
  RandomStdModel,
  TrainingSeqGen,
  FactoredDecodingSeqGen,
  JsonSeqGen,
  TextSeqGen,
  CompiledCrfInstance,
  StaticFeatureManagerBuilder,
  TrainingFactoredFeatureRep,
  RandomLongAlphabet,
  WordProperties,
  StatelessViterbi
}
import org.mitre.jcarafe.util.Options

class CrfSequenceEvaluator(val dCrf: SparseStatelessCrf) extends TrainingUnitEvaluator[MemoryInstanceSequence, CrfWeights, LCrfLossGradient, BasicLCrfSgdUpdater] {
  def this(nls: Int, nfs: Int) = this(new SparseStatelessCrf(nls, nfs))

  def evaluateTrainingUnit(u: MemoryInstanceSequence, params: CrfWeights, updater: BasicLCrfSgdUpdater): LCrfLossGradient = {
    val (ll, sparseGrad) = dCrf.getGradientSingleSequence(u, params.wts, false, false)
    new LCrfLossGradient(1.0, SparseTensor1(dCrf.nfs, sparseGrad.umap))
  }

  def copy(): CrfSequenceEvaluator = new CrfSequenceEvaluator(new SparseStatelessCrf(dCrf.nls, dCrf.nfs))
}

class LCrfLossGradient(l: Double, val gr: SparseTensor1) extends LossGradient[LCrfLossGradient](l) {
  def add(other: LCrfLossGradient) = {
    gr += other.gr
    new LCrfLossGradient(l + other.loss, gr)
  }
  def asArray = { gr.asArray }    
}

class CrfWeights(val wts: Array[Double], m: Float = 1.0f) extends Weights[CrfWeights](m) with Serializable {
  def this(_wts: Array[Double]) = this(_wts, 1.0f)
  
  def compress(): Unit = {}
  def decompress(): Unit = {}
  def weightAt(i: Int) = throw new RuntimeException("Not implemented")
  def compose(otherWeights: CrfWeights) = {
    this *= mass
    otherWeights *= otherWeights.mass
    this ++ otherWeights
    val nmass = mass + otherWeights.mass
    this *= (1.0f / nmass)
    new CrfWeights(this.wts, nmass)
  }

  def add(otherWeights: CrfWeights): CrfWeights = {
    this += otherWeights
    this
  }

  def addEquals(otherWeights: CrfWeights): Unit = {
    var i = 0; while (i < wts.length) {
      wts(i) += otherWeights.wts(i)
      i += 1
    }
  }
  
  def timesEquals(v: Float) = { 
    var i = 0; while (i < wts.length) {
      wts(i) *= v
      i += 1
    }
  }

  def l2norm = throw new RuntimeException("Norm not implemented yet")
  
  def updateFromArray(ar: Array[Float]) = {
    var i = 0; while (i < wts.length) {
      wts(i) = ar(i)
      i += 1
    }
    this
  }
  
  def updateFromArray(ar: Array[Double]) = {
    var i = 0; while (i < wts.length) {
      wts(i) = ar(i)
      i += 1
    }
    this
  }   
  
  def asArray: Array[Float] = Array.tabulate(wts.length)(i => wts(i).toFloat)
  def asTensor1: org.mitre.mandolin.util.Tensor1 = new DenseTensor1(Array.tabulate(wts.length)(i => wts(i).toFloat))
  def copy(): org.mitre.mandolin.lcrf.CrfWeights = new CrfWeights(wts.clone())
  val numWeights: Int = wts.length
}

class BasicLCrfSgdUpdater(val initialLearningRate: Double = 0.2, lambda: Double = 0.1) extends Updater[CrfWeights, LCrfLossGradient, BasicLCrfSgdUpdater] {
  var numIterations = 0
  
  def asArray : Array[Float] = throw new RuntimeException("As array not available for complex updater")
  def updateFromArray(ar: Array[Float]) = throw new RuntimeException("From array not available for complex updater")
  def compress() = this
  def decompress() = this

  def copy() = {
    val sgd = new BasicLCrfSgdUpdater(initialLearningRate)
    sgd.numIterations = this.numIterations
    sgd
  }
  def resetLearningRates(v: Float) = {}
  def compose(u: BasicLCrfSgdUpdater) = this
  def updateWeights(lossGrad: LCrfLossGradient, weights: CrfWeights): Unit = {
    val eta_t = initialLearningRate / (1.0 + (initialLearningRate * numIterations * lambda))
    lossGrad.gr.forEach{case (i,v) =>
      weights.wts(i) += (v * eta_t)
    }
    numIterations += 1
  }
}

class CrfModelWriter(sgen: TrainingSeqGen[String], numFeatures: Int) extends CrfSeqUtils {

  var epoch = 0
  val nls = sgen.getNumberOfStates
  val nfs = numFeatures

  def writeModel(f: java.io.File, w: CrfWeights) : Unit = {
    val core = new CoreModel(w.wts, nfs, nls)
    val m = getRandModel(sgen, false, core)
    StandardSerializer.writeModel(m, f)
  }

}

class CrfModelReader {
  def readModel(f: java.io.File) : CrfWeights = {
    val m = StandardSerializer.readModel(f)
    new CrfWeights(m.crf.params)
  }
}

class CrfEvalPredictor(crfDecoder: StatelessViterbi, dim: Int)
  extends EvalPredictor[MemoryInstanceSequence, CrfWeights, Seq[Int], DiscreteConfusion] with Serializable {

  def getPrediction(unit: MemoryInstanceSequence, weights: CrfWeights): Seq[Int] = {
    crfDecoder.assignBestSequence(unit.iseq, weights.wts)
    unit.iseq map { _.label }
  }

  def getScoredPredictions(unit: MemoryInstanceSequence, weights: CrfWeights): Seq[(Float,Seq[Int])] = {
    throw new RuntimeException("Score predictions unavailable for CRF")
  }

  def getLoss(unit: MemoryInstanceSequence, weights: CrfWeights): Double = {
    0.0 // computing this is a bit expensive, test loss isn't usually necessary
  }

  def getConfusion(unit: MemoryInstanceSequence, weights: CrfWeights): DiscreteConfusion = {
    crfDecoder.assignBestSequence(unit.iseq, weights.wts)
    val predictedSequence = unit.iseq map { e => (e.label, e.orig) }
    DiscreteConfusion(dim, predictedSequence.toVector)
  }
}


trait CrfSettings extends AppSettings {
  val featureSet       = asStr("mandolin.crf.feature-set")
  val inFileMode       = asStr("mandolin.crf.input-mode")
  val noPreProc        = asBoolean("mandolin.crf.no-pre-proc")
  val tagsetFile       = asStr("mandolin.crf.tagset-file")
  val wordPropFile     = asStrOpt("mandolin.crf.word-property-file")
  val lexiconDir       = asStrOpt("mandolin.crf.lexicon")
  val outputFile       = asStrOpt("mandolin.crf.output-file")
}

trait CrfSeqUtils {

  def getStdModel(sgen: TrainingSeqGen[String], begin: Boolean, cm: CoreModel) = {
    new StdModel(sgen.getModelName, begin, sgen.getLexicon, sgen.getWordProps, sgen.getWordScores, sgen.getInducedFeatureMap,
      1, sgen.getLAlphabet, cm, sgen.frep.fsetMap)
  }

  def getRandModel(sgen: TrainingSeqGen[String], begin: Boolean, cm: CoreModel) = {
    new RandomStdModel(sgen.getModelName, begin, sgen.getLexicon, sgen.getWordProps, sgen.getWordScores, sgen.getInducedFeatureMap,
      1, sgen.getLAlphabet, cm, sgen.frep.faMap.asInstanceOf[RandomLongAlphabet])
  }
  
  def getDecodingSeqGen(inMode: String, crfOpts: Options) = {
    val model = StandardSerializer.readModel(crfOpts.model.get)
    inMode match {
      case "json" => new FactoredDecodingSeqGen[String](model, crfOpts) with JsonSeqGen
      case _ => new FactoredDecodingSeqGen[String](model, crfOpts) with TextSeqGen
    }
  }

  def getSeqGen(inMode: String, frep: TrainingFactoredFeatureRep[String], crfOpts: Options) = {
    inMode match {
      case "json" => new TrainingSeqGen[String](frep, crfOpts) with JsonSeqGen
      case "inline" => new TrainingSeqGen[String](frep, crfOpts) with TextSeqGen
      case _ => new TrainingSeqGen[String](frep, crfOpts) with TextSeqGen    
    }
  }
}


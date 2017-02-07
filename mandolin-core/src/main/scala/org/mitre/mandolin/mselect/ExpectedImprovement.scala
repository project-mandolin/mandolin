package org.mitre.mandolin.mselect

import scala.collection.mutable.ArrayBuffer
import WorkPullingPattern._
import org.mitre.mandolin.glp.{ GLPBayesianRegressor, GLPWeights, GLPFactor, GLPTrainerBuilder, LinearLType, 
  LType, InputLType, SparseInputLType, TanHLType }
import org.mitre.mandolin.util.{ StdAlphabet, Alphabet, IdentityAlphabet, DenseTensor1 => DenseVec }
import org.mitre.mandolin.transform.{ FeatureExtractor }
import org.mitre.mandolin.predict.OutputConstructor
import org.mitre.mandolin.glp.{GLPFactor, StdGLPFactor}
import org.mitre.mandolin.predict.local.{ LocalDecoder, LocalTrainer, LocalTrainTester, LocalTrainDecoder, LocalPosteriorDecoder }
import org.slf4j.LoggerFactory

import breeze.linalg.{ DenseVector => BreezeVec, DenseMatrix => BreezeMat }


abstract class AcquisitionFunction {
  def score(configs: Vector[ModelConfig]) : Vector[Double]
  def score(config: ModelConfig) : Double
  def train(evalResults: Seq[ScoredModelConfig]) : Unit
  
}

abstract class ExpectedImprovement extends AcquisitionFunction {

}

class RandomAcquisitionFunction extends AcquisitionFunction {
  def score(config: ModelConfig) : Double = util.Random.nextDouble()
  def train(evalResults: Seq[ScoredModelConfig]) : Unit = {}
  def score(configs: Vector[ModelConfig]) : Vector[Double] = configs map {_ => util.Random.nextDouble()}
}

class MockAcquisitionFunction extends AcquisitionFunction {
  def score(configs: Vector[ModelConfig]) : Vector[Double] = configs map {_ => util.Random.nextDouble()}
  def score(config: ModelConfig) : Double = util.Random.nextDouble()
  def train(evalResults: Seq[ScoredModelConfig]) : Unit = {
    val ms = (util.Random.nextDouble() * 1 * 1000).toLong
    Thread.sleep(ms)
  }
}

trait MetaParameterHandler {
  /**
   * If `nfs` > 0 then this will compute a feature vector using the fixed alphabet
   * If `nfs` <= 0 then this will simply update the alphabet passed in (to build the symbol table)
   */
  protected def configToFeatures(c: ModelConfig, fa: Alphabet, nfs: Int) : Option[DenseVec] = {
    val dv = if (nfs > 0) Some(DenseVec.zeros(nfs)) else None
    c.categoricalMetaParamSet map { cmp =>
      val ss = cmp.getValue.s
      val fid = fa.ofString(cmp.getName+"_"+ss)
      if (fid >= 0) dv foreach { case dv => dv(fid) = 1.0f } // unit features for categorical meta-parameters
    }
    c.realMetaParamSet map {rmp =>
      val fid = fa.ofString(rmp.getName)
      dv foreach { case dv => dv(fid) = rmp.getValue.v.toFloat }
      }
    dv
  }   
}

class AlphabetBuilder extends MetaParameterHandler {
  
  def build(modelSpace: ModelSpace) = {
    val cats = modelSpace.catMPs.toVector
    val reals = modelSpace.realMPs
    val fa = new StdAlphabet
    reals foreach {r =>
      fa.ofString(r.name)}
    cats foreach {c =>
      val s = c.valSet.size
      for (i <- 0 until s - 1) { // this excludes LAST value to avoid dummy encoded categorical variables being perfectly correlated resulting in Singular Matrix
        val ss = c.name+"_"+c.valSet(i)
        fa.ofString(ss)  
      }      
    }
    fa.ensureFixed
    fa
  }
  
  def build(configs: Seq[ScoredModelConfig]) = {
    val fa = new StdAlphabet
    configs foreach {c => configToFeatures(c.mc, fa, -1)}
    fa.ensureFixed
    fa
  }
  
  def addTo(configs: Seq[ScoredModelConfig], fa: Alphabet) = {
    fa.ensureUnFixed // allow alphabet to be updated
    configs foreach {c => configToFeatures(c.mc, fa, -1)}
    fa.ensureFixed
    fa
  }
}

class MetaParameterExtractor(fa: Alphabet, nfs: Int) 
extends FeatureExtractor[ScoredModelConfig, GLPFactor] with MetaParameterHandler with Serializable {
    
  def getAlphabet = fa
  
  def extractFeatures(s: ScoredModelConfig) : GLPFactor = {    
    val lv = s.sc.toFloat
    val targetVec = DenseVec.tabulate(1){_ => lv}
    val features = configToFeatures(s.mc, fa, nfs)
    new StdGLPFactor(features.get, targetVec)  
  }
  def getNumberOfFeatures = nfs
}

class MetaParamModelOutputConstructor extends OutputConstructor[ScoredModelConfig, (Double, Double), (Double, Double)] {
  def constructOutput(input: ScoredModelConfig, response: (Double, Double), tunitStr: String) : (Double, Double) = {
    response
  }
  def responseString(r: (Double,Double)) : String = {
    "Mean: " + r._1 + " Variance: " + r._2 
  }
  def intToResponseString(i: Int) : String = throw new RuntimeException("Invalid method for regressor")
}

class MetaParamDecoder(
    val decoder: LocalDecoder[ScoredModelConfig, GLPFactor, GLPWeights, (Double, Double), (Double, Double)],
    val weights : GLPWeights
    ) {
  def decode(c: ModelConfig) = decoder.run(Vector(ScoredModelConfig(0.0,c)), weights)  
  def decode(c: Vector[ModelConfig]) = decoder.run(c map {cc => ScoredModelConfig(0.0,cc)}, weights)
    
}

/**
 * This implements a simple acquisition function using a MLP (or linear model) with a
 * Bayesian output layer to approxiamte a Gaussian Process
 * Expected Improvement is used as the acquisition function
 * It is currently hard-coded to assume that GREATER is BETTER in terms of evaluations,
 * so use accuracy or area under ROC (not error rate) in evaluators.
 * 
 * @author wellner@mitre.org
 */
class BayesianNNAcquisitionFunction(ms: ModelSpace) extends AcquisitionFunction {
  
  
  private val linear = false

  val mspec : IndexedSeq[LType] = 
    if (linear) IndexedSeq(LType(InputLType), LType(LinearLType))
    else IndexedSeq(LType(InputLType), LType(TanHLType, dim=3, l2 = 0.01f), LType(LinearLType))
    
  val numIterations = 100
  
  var curData : Vector[ScoredModelConfig] = Vector()
  var bestScore : Double = 0.0
  
  val gaussian = breeze.stats.distributions.Gaussian(0.0,1.0) // normal distribution, variance 1.0

  // build the feature alphabet once, up-front
  val fa = (new AlphabetBuilder).build(ms)  
  val fe = new MetaParameterExtractor(fa, fa.getSize)    
    
  var curDecoder : Option[MetaParamDecoder] = None
  
  val log = LoggerFactory.getLogger(getClass)     
  
  private def getPredictiveMeanVariance(config: ModelConfig) : (Double, Double) = {
    curDecoder match {
      case Some(d) => d.decode(config)(0)
      case None => (util.Random.nextDouble(), 0.0)
    }
  }
  
  private def getExpectedImprovement(config: ModelConfig) : Double = {
    val (mu, sigma) = getPredictiveMeanVariance(config)

    if (sigma > 0.0 && (curData.length > 0)) {
      val optimum = bestScore
      val zfactor = (mu - optimum) / sigma
      (mu - optimum) *  gaussian.cdf(zfactor) + sigma * gaussian.pdf(zfactor)
    }
    else 0.0
  }
  
  def score(config: ModelConfig) : Double = {
    getExpectedImprovement(config)    
  }
  
  def score(configs: Vector[ModelConfig]) : Vector[Double] = {
    configs map getExpectedImprovement
  }

  // XXX - can probably just make this static, built up front since features won't change
  def getMetaTrainer = GLPTrainerBuilder(mspec, fe, fa.getSize, 1)    
  
  def train(evalResults: Seq[ScoredModelConfig]) : Unit = {
    // update the data
    curData = evalResults.toVector
    bestScore = curData.maxBy{_.sc}.sc // current best score - larger scores always better
    val (trainer, glp) = getMetaTrainer
    log.info("Number of layers = " + glp.numLayers)
    for (i <- 0 until glp.numLayers) {
      log.info("Dimension layer " + i + " is = " + glp.layers(i).getNumberOfOutputs)
    }
    
    // XXX - should eventually optimize this to avoid recomputing features over entire set of instances each time
    val glpFactors = curData map { trainer.getFe.extractFeatures }
    log.info("Glp factors size = " + glpFactors.length)

    // numIterations should probably be dynamic based on MLP and/or number of data points
    val (weights,_) = trainer.retrainWeights(glpFactors, numIterations)
    
    val dfInVecs = glpFactors map {x =>
      val inV = 
        if (linear) x.getInput 
        else {
          // run the MLP and get penultimate layer
          glp.forwardPass(x.getInput, x.getOutput, weights, false)          
          glp.layers(glp.numLayers - 2).getOutput(false)          
        }
      val v = BreezeVec.tabulate(inV.getDim + 1){i => if (i > 0) inV(i-1).toDouble else 1.0} // add in bias to designmatrix
      BreezeMat(v)
    }
    
    val bMat = dfInVecs.reduce{(a,b) => BreezeMat.vertcat(a,b)}  // the design matrix
    //log.info("Design matrix dims = " + bMat.rows + ", " + bMat.cols)
    val dfArray = glpFactors.toArray
    val targetsVec = BreezeVec.tabulate(glpFactors.length){i => dfArray(i).getOutput(0).toDouble} // the target vector
    val predictor = new GLPBayesianRegressor(glp, bMat, targetsVec, 0.0, 0.0, false)
    val oc = new MetaParamModelOutputConstructor()
    val decoder = new LocalDecoder(trainer.getFe, predictor, oc)
    curDecoder = Some(new MetaParamDecoder(decoder, weights))
  }
}
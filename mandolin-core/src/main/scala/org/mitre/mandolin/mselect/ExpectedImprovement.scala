package org.mitre.mandolin.mselect

import scala.collection.mutable.ArrayBuffer
import WorkPullingPattern._
import org.mitre.mandolin.glp.{ GLPBayesianRegressor, GLPWeights, GLPFactor, GLPTrainerBuilder, LinearLType, 
  LType, InputLType, SparseInputLType, TanHLType }
import org.mitre.mandolin.util.{ AlphabetWithUnitScaling, StdAlphabet, Alphabet, IdentityAlphabet, DenseTensor1 => DenseVec }
import org.mitre.mandolin.transform.{ FeatureExtractor }
import org.mitre.mandolin.predict.OutputConstructor
import org.mitre.mandolin.glp.{GLPFactor, StdGLPFactor}
import org.mitre.mandolin.predict.local.{ LocalDecoder, LocalTrainer, LocalTrainTester, LocalTrainDecoder, LocalPosteriorDecoder }
import org.slf4j.LoggerFactory

import breeze.linalg.{ DenseVector => BreezeVec, DenseMatrix => BreezeMat }



abstract class ScoringFunction {
  def score(configs: Vector[ModelConfig]) : Vector[Double]
  def score(config: ModelConfig) : Double
  def train(evalResults: Seq[ScoredModelConfig]) : Unit  
}


abstract class AcquisitionFunction {
  def score(optimum: Double, mu: Double, variance: Double) : Double
}


class ExpectedImprovement extends AcquisitionFunction {
  val gaussian = breeze.stats.distributions.Gaussian(0.0,1.0)

  def score(optimum: Double, mu: Double, variance: Double) : Double = {

     if(variance > 0.0) {
      val standardDeviation = math.sqrt(variance)
      val z = ((mu - optimum) / standardDeviation)
      val probablilityOfImprovement = gaussian.cdf(z)

      (standardDeviation * z * probablilityOfImprovement) + gaussian.pdf(z)
    } else {
      val z = mu - optimum
      val probablilityOfImprovement = gaussian.cdf(z)
      (z * probablilityOfImprovement) + gaussian.pdf(z)
    }
  }
}

class RandomAcquisition extends AcquisitionFunction {
  def score(optimum: Double, mu: Double, variance: Double) : Double = {
    util.Random.nextDouble()
  }
}

class ExpectedImprovementVer2 extends AcquisitionFunction {

  val gaussian = breeze.stats.distributions.Gaussian(0.0,1.0)

  def score(optimum: Double, mu: Double, variance: Double) : Double = {
    if (variance > 0.0) {
      val zfactor = (mu - optimum) / variance
      (mu - optimum) * gaussian.cdf(zfactor) + variance * gaussian.pdf(zfactor)
    }
    else 0.0
  }
}

class ProbabilityOfImprovement extends AcquisitionFunction {

  val gaussian = breeze.stats.distributions.Gaussian(0.0,1.0)

  def score(optimum: Double, mu: Double, variance: Double) : Double = {
    if(variance > 0.0) {
      val z = ((mu - optimum) / math.sqrt(variance))
      gaussian.cdf(z)
    } else {
      val z = mu - optimum
      gaussian.cdf(z)
    }
  }
}

class UpperConfidenceBound(k: Double) extends AcquisitionFunction {

  def score(optimum: Double, mu: Double, variance: Double) : Double = {
    val standardDeviation = math.sqrt(variance)
    mu + k * standardDeviation
  }
}

class RandomScoringFunction extends ScoringFunction {
  def score(config: ModelConfig) : Double = util.Random.nextDouble()
  def train(evalResults: Seq[ScoredModelConfig]) : Unit = {}
  def score(configs: Vector[ModelConfig]) : Vector[Double] = configs map {_ => util.Random.nextDouble()}
}

class MockScoringFunction extends ScoringFunction {
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
    c.categoricalMetaParamSet foreach { cmp =>
      val ss = cmp.getValue.s
      val fid = fa.ofString(cmp.getName+"_"+ss)
      if (fid >= 0) dv foreach { case dv => dv(fid) = 1.0f } // unit features for categorical meta-parameters
    }
    c.realMetaParamSet foreach {rmp =>
      val fid = fa.ofString(rmp.getName)
      dv foreach { case dv => dv(fid) = fa.getValue(fid, rmp.getValue.v).toFloat }
      }
    c.intMetaParamSet foreach { imp => 
      val fid = fa.ofString(imp.getName)
      dv foreach { case dv => dv(fid) = fa.getValue(fid,imp.getValue.v).toFloat }
      }
    
    c.topo foreach { topo =>
      val numLayers = topo.length
      val fidLayers = fa.ofString("num_layers")
      dv foreach {dv => dv(fidLayers) = fa.getValue(fidLayers,numLayers).toFloat }
      // now number of total weights
      var totalWeights = 0
      for (i <- 0 until topo.length) {
        val n = if (i == 0) topo(i).dim * c.inDim else topo(i).dim * topo(i-1).dim
        totalWeights += n        
      }
      totalWeights += (topo(topo.length - 1).dim * c.outDim)
      val fidTotalWeight = fa.ofString("total_weights")
      dv foreach {dv => dv(fidTotalWeight) = fa.getValue(fidTotalWeight, totalWeights).toFloat }
    }

    dv
  }   
}

class AlphabetBuilder extends MetaParameterHandler {
  
  def build(modelSpace: ModelSpace) = {
    val cats = modelSpace.catMPs
    val reals = modelSpace.realMPs
    val ints = modelSpace.intMPs
    val fa = new AlphabetWithUnitScaling
    reals foreach {r =>
      fa.ofString(r.name, r.vs.lower) // add lowest value
      fa.ofString(r.name, r.vs.upper) // add highest
      }
    ints foreach { r => 
      fa.ofString(r.name, r.vs.lower)
      fa.ofString(r.name, r.vs.upper)
    }
    cats foreach {c =>
      val s = c.valSet.size
      for (i <- 0 until s - 1) { // this excludes LAST value to avoid dummy encoded categorical variables being perfectly correlated resulting in Singular Matrix
        val ss = c.name+"_"+c.valSet(i)
        fa.ofString(ss)  
      }      
    }
    modelSpace.topoMPs foreach { topoSpace =>
      val items = topoSpace.liSet.li
      var minLayers = 100
      var maxLayers = 0
      var minWeights = Integer.MAX_VALUE
      var maxWeights = 0
      items foreach {it =>
        it.valueSet match {
          case SetValue(x) => 
            if (x.size < minLayers) minLayers = x.size
            if (x.size > maxLayers) maxLayers = x.size
            var localMinWeights = 0
            var localMaxWeights = 0
            if (x.length > 0) {
              for (i <- 0 until x.length) {                
                val (l1min, l1max) = x(i).valueSet match {
                  case TupleSet4(a,b,c,d) => b.valueSet match {
                    case x: IntSet => (x.lower, x.upper)
                  }                     
                }
                if (i == 0) {
                  localMinWeights += modelSpace.idim * l1min
                  localMaxWeights += modelSpace.idim * l1max
                } else {
                  val (l0min, l0max) = x(i-1).valueSet match {
                    case TupleSet4(a,b,c,d) => b.valueSet match {
                      case x: IntSet => (x.lower,x.upper)                      
                    }
                   }
                  localMinWeights += l1min * l0min
                  localMaxWeights += l1max * l0max
                }          
                if (i == x.length - 1) { // weights for last hidden to output layer connection
                  localMinWeights += l1min * modelSpace.odim
                  localMaxWeights += l1max * modelSpace.odim
                }
              }
            } else { // just for a linear model
              localMinWeights = modelSpace.idim * modelSpace.odim
            }
            if (localMinWeights < minWeights) minWeights = localMinWeights
            if (localMaxWeights > maxWeights) maxWeights = localMaxWeights
        }
        }
      // set minimum and maximum values of these features (for scaling)
      fa.ofString("num_layers", minLayers)
      fa.ofString("num_layers", maxLayers)
      fa.ofString("total_weights", minWeights)
      fa.ofString("total_weights", maxWeights)
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
class BayesianNNScoringFunction(ms: ModelSpace, acqFunc: AcquisitionFunction = new ExpectedImprovement) extends ScoringFunction {
  
  
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
  
  private def calculateScore(config: ModelConfig) : Double = {
    val (mu, variance) = getPredictiveMeanVariance(config)
    acqFunc.score(bestScore, mu, variance)
  }

  def score(config: ModelConfig) : Double = {
    calculateScore(config)
  }

  def score(configs: Vector[ModelConfig]) : Vector[Double] = {
    configs map calculateScore
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
  
  
  // XXX - can probably just make this static, built up front since features won't change
  def getMetaTrainer = GLPTrainerBuilder(mspec, fe, fa.getSize, 1)    
  
  def train(evalResults: Seq[ScoredModelConfig]) : Unit = {
    // update the data
    curData = evalResults.toVector
    bestScore = curData.maxBy{_.sc}.sc // current best score - LARGER! scores always better here
    val (trainer, glp) = getMetaTrainer
    log.info("Number of layers = " + glp.numLayers)
    for (i <- 0 until glp.numLayers) {
      log.info("Dimension layer " + i + " is = " + glp.layers(i).getNumberOfOutputs)
    }
    
    // XXX - should eventually optimize this to avoid recomputing features over entire set of instances each time
    val glpFactors = curData map { trainer.getFe.extractFeatures }

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
package org.mitre.mandolin.mselect

import scala.collection.mutable.ArrayBuffer
import org.mitre.mandolin.glp.{ANNetwork, GLPBayesianRegressor, GLPWeights, GLPTrainerBuilder, LinearLType,
LType, InputLType, TanHLType}
import org.mitre.mandolin.util.{AlphabetWithUnitScaling, StdAlphabet, Alphabet, DenseTensor1 => DenseVec}
import org.mitre.mandolin.transform.{FeatureExtractor}
import org.mitre.mandolin.predict.OutputConstructor
import org.mitre.mandolin.glp.{GLPFactor, StdGLPFactor}
import org.mitre.mandolin.predict.local.LocalDecoder
import org.slf4j.LoggerFactory

import breeze.linalg.{DenseVector => BreezeVec, DenseMatrix => BreezeMat}


abstract class ScoringFunction {
  def scoreConcurrent(configs: Vector[ModelConfig], n: Int): Vector[(Double, ModelConfig)]

  def score(config: ModelConfig): Double

  def train(evalResults: Seq[ScoredModelConfig]): Unit

  def scoreWithNewK(config: ModelConfig, K: BreezeMat[Double]): Double

  def getUpdatedK(bm: Vector[ModelConfig]): Option[BreezeMat[Double]]
}


abstract class AcquisitionFunction {
  def score(optimum: Double, mu: Double, variance: Double): Double

  val mixParam: Double = 0.0
}


class ExpectedImprovement extends AcquisitionFunction {
  val gaussian = breeze.stats.distributions.Gaussian(0.0, 1.0)

  def score(optimum: Double, mu: Double, variance: Double): Double = {

    if (variance > 0.0) {
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
  def score(optimum: Double, mu: Double, variance: Double): Double = {
    util.Random.nextDouble()
  }
}

class ExpectedImprovementVer2 extends AcquisitionFunction {

  val gaussian = breeze.stats.distributions.Gaussian(0.0, 1.0)

  def score(optimum: Double, mu: Double, variance: Double): Double = {
    if (variance > 0.0) {
      val stdDev = math.sqrt(variance)
      val zfactor = (mu - optimum) / stdDev
      (mu - optimum) * gaussian.cdf(zfactor) + stdDev * gaussian.pdf(zfactor)
    }
    else 0.0
  }
}

class ProbabilityOfImprovement extends AcquisitionFunction {

  val gaussian = breeze.stats.distributions.Gaussian(0.0, 1.0)

  def score(optimum: Double, mu: Double, variance: Double): Double = {
    if (variance > 0.0) {
      val z = ((mu - optimum) / math.sqrt(variance))
      gaussian.cdf(z)
    } else {
      val z = mu - optimum
      gaussian.cdf(z)
    }
  }
}

class UpperConfidenceBound(k: Double) extends AcquisitionFunction {

  // make this accessible 
  override val mixParam = k

  def score(optimum: Double, mu: Double, variance: Double): Double = {
    val standardDeviation = math.sqrt(variance)
    mu + k * standardDeviation
  }
}



class MockScoringFunction extends ScoringFunction {

  def scoreConcurrent(configs: Vector[ModelConfig], n: Int): Vector[(Double, ModelConfig)] = Vector()

  def score(config: ModelConfig): Double = util.Random.nextDouble()

  def train(evalResults: Seq[ScoredModelConfig]): Unit = {
    val ms = (util.Random.nextDouble() * 1 * 1000).toLong
    Thread.sleep(ms)
  }

  def scoreWithNewK(config: ModelConfig, K: BreezeMat[Double]) : Double = throw new RuntimeException("Unimplemented")
  def getUpdatedK(bm: Vector[ModelConfig]) : Option[BreezeMat[Double]] = None
}

class RandomScoringFunction extends ScoringFunction {
  
  def scoreConcurrent(configs: Vector[ModelConfig], n: Int) : Vector[(Double, ModelConfig)] = configs map {c => (util.Random.nextDouble(), c)}
  def score(config: ModelConfig) : Double = util.Random.nextDouble()
  def train(evalResults: Seq[ScoredModelConfig]) : Unit = {}
  def scoreWithNewK(config: ModelConfig, K: BreezeMat[Double]) : Double = throw new RuntimeException("Unimplemented")
  def getUpdatedK(bm: Vector[ModelConfig]) : Option[BreezeMat[Double]] = None
}

trait MetaParameterHandler {
  /**
    * If `nfs` > 0 then this will compute a feature vector using the fixed alphabet
    * If `nfs` <= 0 then this will simply update the alphabet passed in (to build the symbol table)
    */
  protected def configToFeatures(c: ModelConfig, fa: Alphabet, nfs: Int): Option[DenseVec] = {
    val dv = if (nfs > 0) Some(DenseVec.zeros(nfs)) else None
    c.categoricalMetaParamSet foreach { cmp =>
      val ss = cmp.getValue.s
      val fid = fa.ofString(cmp.getName + "_" + ss)
      if (fid >= 0) dv foreach { case dv => dv(fid) = 1.0f } // unit features for categorical meta-parameters
    }
    c.realMetaParamSet foreach { rmp =>
      val fid = fa.ofString(rmp.getName)
      dv foreach { case dv => dv(fid) = fa.getValue(fid, rmp.getValue.v).toFloat }
    }
    c.intMetaParamSet foreach { imp =>
      val fid = fa.ofString(imp.getName)
      dv foreach { case dv => dv(fid) = fa.getValue(fid, imp.getValue.v).toFloat }
    }
    // now add in budget/number of iterations
    if (c.budget > 0) {
      val budgetFid = fa.ofString("budget")
      dv foreach { dv => dv(budgetFid) = fa.getValue(budgetFid, c.budget).toFloat }
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
    reals foreach { r =>
      fa.ofString(r.name, r.vs.lower) // add lowest value
      fa.ofString(r.name, r.vs.upper) // add highest
    }
    ints foreach { r =>
      fa.ofString(r.name, r.vs.lower)
      fa.ofString(r.name, r.vs.upper)
    }
    cats foreach { c =>
      val s = c.valSet.size
      for (i <- 0 until s - 1) {
        // this excludes LAST value to avoid dummy encoded categorical variables being perfectly correlated resulting in Singular Matrix
        val ss = c.name + "_" + c.valSet(i)
        fa.ofString(ss)
      }
    }
    // budget feature - i.e. number of iterations
    fa.ofString("budget", modelSpace.maxBudget.toDouble)
    fa.ofString("budget", 0.0)
    fa.ensureFixed
    fa
  }

  def build(configs: Seq[ScoredModelConfig]) = {
    val fa = new StdAlphabet
    configs foreach { c => configToFeatures(c.mc, fa, -1) }
    fa.ensureFixed
    fa
  }

  def addTo(configs: Seq[ScoredModelConfig], fa: Alphabet) = {
    fa.ensureUnFixed // allow alphabet to be updated
    configs foreach { c => configToFeatures(c.mc, fa, -1) }
    fa.ensureFixed
    fa
  }
}

class MetaParameterExtractor(fa: Alphabet, nfs: Int)
  extends FeatureExtractor[ScoredModelConfig, GLPFactor] with MetaParameterHandler with Serializable {

  def getAlphabet = fa

  def extractFeatures(s: ScoredModelConfig): GLPFactor = {
    val lv = s.sc.toFloat
    val targetVec = DenseVec.tabulate(1) { _ => lv }
    val features = configToFeatures(s.mc, fa, nfs)
    new StdGLPFactor(features.get, targetVec)
  }

  def getNumberOfFeatures = nfs
}

class TimingMetaParameterExtractor(fa: Alphabet, nfs: Int)
  extends MetaParameterExtractor(fa, nfs) {

  override def extractFeatures(s: ScoredModelConfig): GLPFactor = {
    val lv = s.t.toFloat
    val targetVec = DenseVec.tabulate(1) { _ => lv }
    val features = configToFeatures(s.mc, fa, nfs)
    new StdGLPFactor(features.get, targetVec)
  }

}

class MetaParamModelOutputConstructor extends OutputConstructor[ScoredModelConfig, (Double, Double), (Double, Double)] {
  def constructOutput(input: ScoredModelConfig, response: (Double, Double), tunitStr: String): (Double, Double) = {
    response
  }

  def responseString(r: (Double, Double)): String = {
    "Mean: " + r._1 + " Variance: " + r._2
  }

  def intToResponseString(i: Int): String = throw new RuntimeException("Invalid method for regressor")
}

class MetaParamDecoder(
                        val decoder: LocalDecoder[ScoredModelConfig, GLPFactor, GLPWeights, (Double, Double), (Double, Double)],
                        val weights: GLPWeights
                      ) {
  def decode(c: ModelConfig) = decoder.run(Vector(ScoredModelConfig(0.0, 0, c)), weights)

  def decode(c: Vector[ModelConfig]) = decoder.run(c map { cc => ScoredModelConfig(0.0, 0, cc) }, weights)

}

class BayesianNNScoringFunctionWTiming(ms: ModelSpace, acqFunc: AcquisitionFunction = new ExpectedImprovement, numConcurrent: Int = 1) extends BayesianNNScoringFunction(ms, acqFunc, numConcurrent) {
  override val fe = new TimingMetaParameterExtractor(fa, fa.getSize)
}

/**
  * This implements a simple acquisition function using a MLP (or linear model) with a
  * Bayesian output layer to approximate a Gaussian Process
  * Expected Improvement is used as the acquisition function
  * It is currently hard-coded to assume that GREATER is BETTER in terms of evaluations,
  * so use accuracy or area under ROC (not error rate) in evaluators.
  *
  * @author wellner@mitre.org
  */
class BayesianNNScoringFunction(ms: ModelSpace, acqFunc: AcquisitionFunction = new ExpectedImprovement, numConcurrent: Int = 1)
  extends ScoringFunction {

  private val linear = false
  private val useCache = numConcurrent > 1

  val maxIterations = 100

  /**
    * Hold a cache of the input data for use with concurrent acquisition functions
    */
  var dataCache: BreezeMat[Double] = BreezeMat(0.0)
  var bestScore: Double = 0.0

  val gaussian = breeze.stats.distributions.Gaussian(0.0, 1.0) // normal distribution, variance 1.0

  // build the feature alphabet once, up-front
  val fa = (new AlphabetBuilder).build(ms)
  val fe = new MetaParameterExtractor(fa, fa.getSize)

  def getMspec(n: Int): IndexedSeq[LType] = {
    val dim = math.min(fa.getSize * 2, math.max(3, n / 10))
    if (linear) IndexedSeq(LType(InputLType), LType(LinearLType))
    else IndexedSeq(LType(InputLType),
        LType(TanHLType, dim=dim, l2 = 0.001f, maxNorm = 10.0f), 
        LType(TanHLType, dim=dim, l2 = 0.001f, maxNorm = 10.0f), LType(LinearLType))
  }

  var curDecoder: Option[MetaParamDecoder] = None
  var curBayesRegressor: Option[GLPBayesianRegressor] = None
  var curWeights: Option[GLPWeights] = None

  val log = LoggerFactory.getLogger(getClass)

  private def getPredictiveMeanVariance(config: ModelConfig): (Double, Double) = {
    curDecoder match {
      case Some(d) => d.decode(config)(0)
      case None => (util.Random.nextDouble(), 0.0)
    }
  }

  private def calculateScore(config: ModelConfig): Double = {
    val (mu, variance) = getPredictiveMeanVariance(config)
    acqFunc.score(bestScore, mu, variance)
  }

  def getUpdatedK(bm: Vector[ModelConfig]): Option[BreezeMat[Double]] = {
    if (curBayesRegressor.isDefined && curWeights.isDefined) {
      val orig = dataCache
      val regressor = curBayesRegressor.get
      val vecs = bm map { c =>
        val fv = fe.extractFeatures(ScoredModelConfig(0.0, 0, c))
        val bv = regressor.getBasisVector(fv, curWeights.get)
        bv
      }
      vecs foreach { bv => BreezeMat.vertcat(dataCache, BreezeMat(bv)) }
      val kInv1 = regressor.getKInv(dataCache)
      dataCache = orig
      Some(kInv1)
    } else None
  }

  def scoreWithNewK(config: ModelConfig, K: BreezeMat[Double]): Double = {
    // calculateScore(config)
    val regressor = curBayesRegressor.get
    val wts = curWeights.get
    val fv = fe.extractFeatures(ScoredModelConfig(0.0, 0, config))
    val bv = regressor.getBasisVector(fv, wts)
    val (mu, _) = regressor.getPrediction(bv, wts)
    val nvar = bv.t * K * bv + regressor.variance
    acqFunc.score(bestScore, mu, nvar)
  }

  def score(config: ModelConfig): Double = calculateScore(config)

  /**
    * This will return the top N configs to use next assumign they will be evaluated concurrently.
    * XXX - Generalize this to take in a separate vector of configurations already being evaluated to pin down variance
    */
  def scoreConcurrent(configs: Vector[ModelConfig], n: Int): Vector[(Double, ModelConfig)] = {
    val totalSize = configs.size
    val origData = dataCache
    if (curBayesRegressor.isDefined && curWeights.isDefined) {
      log.info("*** Concurrent Scoring **** n = " + n)

      val buf = new ArrayBuffer[(Double, ModelConfig)]
      val regressor = curBayesRegressor.get
      val wts = curWeights.get
      /*
      log.info("Number of pending configurations = " + pendingConfigs.length)
      pendingConfigs foreach { c =>
        val fv = fe.extractFeatures(ScoredModelConfig(0.0,c))
        val bv = regressor.getBasisVector(fv, wts)
        dataCache = BreezeMat.vertcat(dataCache, BreezeMat(bv))
        }
      val kInv1 = regressor.getKInv(dataCache)
      *
      */
      val initialScoredBasisVecs = configs map { c =>
        val fv = fe.extractFeatures(ScoredModelConfig(0.0, 0, c))
        val bv = regressor.getBasisVector(fv, wts)
        val (mu, v) = regressor.getPrediction(bv, wts)
        // don't use predictor variance - instead use updated variance based on pending evaluations
        // val nvar = bv.t * kInv1 * bv + regressor.variance
        val sd = math.sqrt(v)
        val sc = mu + sd //   acqFunc.score(bestScore, mu, v)
        (sc, bv, c)
      }
      val sortedBasisVecs = initialScoredBasisVecs.sortBy {
        _._1
      }.reverse.take(totalSize / 4) // just take top quarter
      val best = sortedBasisVecs(0)
      var pointsInRegion = sortedBasisVecs.toList
      val initialMatVec = BreezeMat(best._2)
      log.info("initialMatVec dims = " + initialMatVec.rows + ", " + initialMatVec.cols)
      log.info("dataCache dims = " + dataCache.rows + ", " + dataCache.cols)

      // buf append ((best._1, best._3))
      // XXX - this is asking for optimizations ..
      //   + keep a priority queue
      //   + since variances are monotonically decreasing, if we find a score (in descending order) less than the current best
      //     --> we are done
      for (i <- 1 until n) {
        // number of concurrent evaluations
        val newKInv = regressor.getKInv(dataCache)
        val scored = pointsInRegion map { case (_, bv, c) =>
          val nvar = bv.t * newKInv * bv + regressor.variance
          (math.sqrt(nvar), bv, c)
        }
        //val inbest = scored.maxBy{_._1}
        val (inbest, sorted) = scored.sortBy {
          _._1
        }.reverse match {
          case h :: t => (h, t)
        }
        // add inbest to set to evaluate
        buf append ((inbest._1, inbest._3))
        dataCache = BreezeMat.vertcat(dataCache, BreezeMat(inbest._2)) // update cache
        pointsInRegion = sorted
      }
      dataCache = origData // set the cache back to
      buf.toVector
    } else {
      val rnd = new util.Random
      configs map { c => (rnd.nextDouble(), c) }
    }
  }

  private def mapInputToBasisVec(x: GLPFactor, glp: ANNetwork, weights: GLPWeights): BreezeMat[Double] = {
    val inV =
      if (linear) x.getInput
      else {
        // run the MLP and get penultimate layer
        glp.forwardPass(x.getInput, x.getOutput, weights, false)
        glp.layers(glp.numLayers - 2).getOutput(false)
      }
    // check for malformed input
    val vv = inV.asArray
    var malformed = false
    var i = 0;
    while (i < vv.length) {
      val a = vv(i)
      if (a.isNaN()) malformed = true
      i += 1
    }
    if (malformed) {
      val buf = new StringBuilder
      val ss = x.getInput.asArray
      var i = 0;
      while (i < ss.length) {
        buf append " "
        buf append (ss(i).toString)
        i += 1
      }
      throw new RuntimeException("NaN in neural net output with input: " + buf.toString())
    }
    val v = BreezeVec.tabulate(inV.getDim + 1){i =>
      if (i > 0) { val v = inV(i-1).toDouble; if (v > 0.0) v - math.random * 0.01 else v + math.random * 0.01 } 
      else 1.0} // add in bias to designmatrix
    BreezeMat(v)
  }

  /**
    * This method expects ALL evaluationResults thus far to be passed in; it does not maintain a cache of evaluations.
    */
  def train(evalResults: Seq[ScoredModelConfig]): Unit = {
    if (evalResults.length > 6) {
      val curData = evalResults.toVector
      bestScore = curData.maxBy {
        _.sc
      }.sc // current best score - LARGER! scores always better here
      val mspec = getMspec(curData.length)
      val (trainer, glp) = GLPTrainerBuilder(mspec, fe, fa.getSize, 1,
        Seq(("mandolin.trainer.optimizer.initial-learning-rate", 0.1),
          ("mandolin.trainer.optimizer.method", "adagrad")))
      log.info("Number of layers = " + glp.numLayers)
      for (i <- 0 until glp.numLayers) {
        log.info("Dimension layer " + i + " is = " + glp.layers(i).getNumberOfOutputs)
      }

      // XXX - should eventually optimize this to avoid recomputing features over entire set of instances each time
      val glpFactors = curData map {
        trainer.getFe.extractFeatures
      }

      val numIterations = maxIterations // math.min(maxIterations, curData.length * 2)
      val (weights, _) = trainer.retrainWeights(glpFactors, numIterations)

      val dfInVecs = glpFactors map { x => mapInputToBasisVec(x, glp, weights) }

      val bMat = dfInVecs.reduce { (a, b) => BreezeMat.vertcat(a, b) } // the design matrix

      log.info("Design matrix dims = " + bMat.rows + ", " + bMat.cols)
      val dfArray = glpFactors.toArray
      val targetsVec = BreezeVec.tabulate(glpFactors.length) { i => dfArray(i).getOutput(0).toDouble } // the target vecto

      val predictor = new GLPBayesianRegressor(glp, bMat, targetsVec, 0.0, 0.0, false)
      val oc = new MetaParamModelOutputConstructor()
      val decoder = new LocalDecoder(trainer.getFe, predictor, oc)
      if (useCache) {
        log.info(" ++++ Setting data cache, weights and regressor for prediction")
        dataCache = bMat
        curWeights = Some(weights)
        curBayesRegressor = Some(predictor)
      }
      curDecoder = Some(new MetaParamDecoder(decoder, weights))
    }
  }
}
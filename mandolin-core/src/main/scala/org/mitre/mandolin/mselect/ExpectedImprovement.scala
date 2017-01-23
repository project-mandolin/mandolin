package org.mitre.mandolin.mselect

import scala.collection.mutable.ArrayBuffer
import WorkPullingPattern._
import org.mitre.mandolin.glp.{ GLPBayesianRegressor, GLPWeights, GLPFactor, GLPTrainerBuilder, LType, InputLType, SparseInputLType, SoftMaxLType }
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
      dv foreach { case dv => dv(fid) = 1.0f } // unit features for categorical meta-parameters
    }
    c.realMetaParamSet map {rmp =>
      val fid = fa.ofString(rmp.getName)
      dv foreach { case dv => dv(fid) = rmp.getValue.v.toFloat }
      }
    dv
  }   
}

class AlphabetBuilder extends MetaParameterHandler {
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

class BayesianNNAcquisitionFunction extends AcquisitionFunction {
  
  val mspec : IndexedSeq[LType] = IndexedSeq(LType(InputLType), LType(SoftMaxLType))
  val curData = new ArrayBuffer[ScoredModelConfig]
  val alphabetBuilder = new AlphabetBuilder
  var curDecoder : Option[MetaParamDecoder] = None
  
  val log = LoggerFactory.getLogger(getClass)
    
  
  def score(config: ModelConfig) : Double = {
    curDecoder match {
      case Some(d) => d.decode(config)(0)._1
      case None => util.Random.nextDouble()
    }
  }
  
  def score(configs: Vector[ModelConfig]) : Vector[Double] = {
    curDecoder match {
      case Some(d) => d.decode(configs) map {_._1}
      case None => configs map {_ => util.Random.nextDouble()}
    }
  }
    
  def getMetaTrainer = {
    val fa = alphabetBuilder.build(curData.toSeq)
    fa.ensureFixed
    val fe = new MetaParameterExtractor(fa, fa.getSize)    
    GLPTrainerBuilder(mspec, fe, fa.getSize, 1)
  }    
  
  def train(evalResults: Seq[ScoredModelConfig]) : Unit = {
    // update the data
    curData ++= evalResults
    val (trainer, glp) = getMetaTrainer
    val glpFactors = curData.toVector map { trainer.getFe.extractFeatures }
    log.info("Glp factors size = " + glpFactors.length)
    println("BayesianNN: Glp factors size = " + glpFactors.length)
    val (weights,_) = trainer.retrainWeights(glpFactors)
    val dfInVecs = glpFactors map {x =>
      val inV = x.getInput    
      val v = BreezeVec.tabulate(inV.getDim + 1){i => if (i > 0) inV(i-1).toDouble else 1.0} // add in bias to designmatrix
      BreezeMat(v)
    }
    val bMat = dfInVecs.reduce{(a,b) => BreezeMat.vertcat(a,b)}  // the design matrix
    log.info("Design matrix dims = " + bMat.rows + ", " + bMat.cols)
    println("Design matrix dims = " + bMat.rows + ", " + bMat.cols)
    val dfArray = glpFactors.toArray
    val targetsVec = BreezeVec.tabulate(glpFactors.length){i => dfArray(i).getOutput(0).toDouble} // the target vector
    val predictor = new GLPBayesianRegressor(glp, bMat, targetsVec, 0.8, 0.0, false)
    val oc = new MetaParamModelOutputConstructor()
    val decoder = new LocalDecoder(trainer.getFe, predictor, oc)
    curDecoder = Some(new MetaParamDecoder(decoder, weights))
  }
}
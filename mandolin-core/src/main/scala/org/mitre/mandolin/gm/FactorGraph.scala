package org.mitre.mandolin.gm

import org.mitre.mandolin.mlp._
import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.util.{Alphabet, IdentityAlphabet, StdAlphabet, DenseTensor1 => DenseVec, SparseTensor1 => SparseVec, Tensor1 => Vec}
import org.mitre.mandolin.mlp.standalone.MMLPOptimizer
import org.mitre.mandolin.optimize.standalone.OnlineOptimizer
import org.mitre.mandolin.predict.standalone.Trainer
import org.slf4j.LoggerFactory

case class AlphabetSet(sa: Alphabet, fa: Alphabet, sla: Alphabet, fla: Alphabet)

object PairWiseOptimizer {

  def getOptimizer(fgSettings: FactorGraphSettings, singleNN: ANNetwork, pairNN: ANNetwork) = {
    val weightsSingle = singleNN.generateRandomWeights
    val sumSquaredSingle = singleNN.generateZeroedLayout
    val weightsPair = pairNN.generateRandomWeights
    val sumSquaredPair = pairNN.generateZeroedLayout

    sumSquaredPair set fgSettings.initialLearnRate // set to the initial learning rate
    sumSquaredSingle set fgSettings.initialLearnRate // set to the initial learning rate
    val uSingle = new MMLPAdaGradUpdater(sumSquaredSingle, fgSettings.initialLearnRate)
    val uPair = new MMLPAdaGradUpdater(sumSquaredPair, fgSettings.initialLearnRate)
    val updater = new MultiFactorAdaGradUpdater(uSingle, uPair, fgSettings.initialLearnRate)
    val evaluator = new PairwiseFactorEvaluator[MultiFactorAdaGradUpdater](singleNN, pairNN)
    val weights = new MultiFactorWeights(weightsSingle, weightsPair, 1.0f)
    new OnlineOptimizer[MultiFactor, MultiFactorWeights, MultiFactorLossGradient, MultiFactorAdaGradUpdater](weights, evaluator, updater, fgSettings)
  }
}

class FactorGraphTrainer(fgSettings: FactorGraphSettings, factorGraph: FactorGraph) {

  val logger = LoggerFactory.getLogger(this.getClass)

  val (singletonNN, factorNN) = getComponents(fgSettings, factorGraph.alphabets)
  val sOpt = MMLPOptimizer.getOptimizer(fgSettings, singletonNN)
  // val fOpt = LocalGLPOptimizer.getLocalOptimizer(fgSettings, factorNN)
  val fOpt = PairWiseOptimizer.getOptimizer(fgSettings, singletonNN, factorNN)
  val ie = new IdentityExtractor(new IdentityAlphabet(1))

  val sTrainer = new Trainer[SingletonFactor, MMLPFactor, MMLPWeights](ie, sOpt)
  val fTrainer = new Trainer[MultiFactor, MultiFactor, MultiFactorWeights](fOpt)

  def getComponents(fgSettings: FactorGraphSettings, as: AlphabetSet) = {
    val singletonSp = ANNBuilder.getMMLPSpec(fgSettings.netspec, as.sa.getSize, as.sla.getSize)
    val factorSp = ANNBuilder.getMMLPSpec(fgSettings.factorSpec, as.fa.getSize, as.fla.getSize)
    //val singletonAnn = ANNetwork(IndexedSeq(LType(InputLType, as.sa.getSize), LType(SoftMaxLType,as.sla.getSize)))
    //val factorAnn    = ANNetwork(IndexedSeq(LType(InputLType, as.fa.getSize), LType(SoftMaxLType,as.fla.getSize)))
    (ANNetwork(singletonSp), ANNetwork(factorSp))
  }

  def trainModels() = {
    val t = System.nanoTime
    logger.info("Estimating singleton parameters ... with " + fgSettings.numEpochs + " epochs")
    val (sWeights, _) = sTrainer.trainWeights(factorGraph.singletons)
    logger.info("Estimation finished in " + ((System.nanoTime - t) / 1E9) + " seconds ")
    val fm = if (factorGraph.factors.length > 0) {
      logger.info("Estimating non-singular factor parameters ... with " + fgSettings.numEpochs + " epochs" )
      val t1 = System.nanoTime

      val (fWeights,_) = fTrainer.retrainWeights(factorGraph.factors, fgSettings.numEpochs)

      logger.info("Estimation finished in " + ((System.nanoTime - t1) / 1E9) + " seconds ")
      new PairFactorModel(singletonNN, factorNN, fWeights)
    } else {
      val emptyWeights = new MultiFactorWeights(new MMLPWeights(new MMLPLayout(IndexedSeq())), new MMLPWeights(new MMLPLayout(IndexedSeq())), 1.0f)
      new PairFactorModel(singletonNN, factorNN, emptyWeights)
    }
    val sm = new SingletonFactorModel(new CategoricalMMLPPredictor(singletonNN, true), sWeights)
    new TrainedFactorGraph(fm, sm, factorGraph, fgSettings.sgAlpha)
  }
}

class FactorGraph(val factors: Vector[MultiFactor], val singletons: Vector[SingletonFactor], val alphabets: AlphabetSet)

class TrainedFactorGraph(val factorModel: PairFactorModel, val singletonModel: SingletonFactorModel,
                         _fs: Vector[MultiFactor], _ss: Vector[SingletonFactor], _a: AlphabetSet, init: Double = 0.1,
                         inferType: String = "star")
  extends FactorGraph(_fs, _ss, _a) {
  def this(fm: PairFactorModel, sm: SingletonFactorModel, fg: FactorGraph, lr: Double) = this(fm, sm, fg.factors, fg.singletons, fg.alphabets, lr)

  def this(fm: PairFactorModel, sm: SingletonFactorModel, fg: FactorGraph, lr: Double, inferType: String) = this(fm, sm, fg.factors, fg.singletons, fg.alphabets, lr, inferType)

  //val inference = new SubgradientInference(factorModel, singletonModel, init)
  val inference = inferType match {
    case "subgrad" => new SubgradientInference(factorModel, singletonModel, init)
    case "smoothgrad" => new SmoothedGradientInference(factorModel, singletonModel, init)
    case _ => new StarCoordinatedBlockMinimizationInference(factorModel, singletonModel, init)
  }
  //val inference = new SmoothedGradientInference(factorModel, singletonModel, init)

  def mapInfer(n: Int) = {
    inference.mapInfer(factors, singletons, n)
  }

  def getMap() = {
    singletons map { s => s.getMode(singletonModel, true) }
  }

  def getAccuracy = {
    val cor = singletons.foldLeft(0) { case (ac, v) => if (v.label == v.getMode(singletonModel, true)) ac + 1 else ac }
    cor.toDouble / singletons.length
  }

  def renderMapOutput(ofile: String, la: Alphabet) = {
    val o = new java.io.PrintWriter(new java.io.File(ofile))
    val invLa = la.getInverseMapping
    singletons foreach { v =>
      o.print(v.getInput.getUniqueKey.getOrElse("?"))
      o.print(' ')
      o.print(invLa(v.getMode(singletonModel, true)))
      o.print('\n')
    }
    o.close
  }
}

object FactorGraph {

  val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)

  def getLineInfo(rest: List[String], alphabet: Alphabet, buildVecs: Boolean) = {
    val fvbuf = new collection.mutable.ArrayBuffer[Feature]
    rest foreach { f =>
      f.split(':').toList match {
        case a :: b :: Nil =>
          if (buildVecs) {
            val bd = b.toDouble
            val id = alphabet.ofString(a, bd)
            if (id >= 0) fvbuf append (new NonUnitFeature(alphabet.ofString(a, bd), bd))
          } else alphabet.ofString(a)
        case a :: Nil =>
          if (buildVecs) {
            val id = alphabet.ofString(a)
            if (id >= 0) fvbuf append (new Feature(id))
          } else alphabet.ofString(a)
        case a => throw new RuntimeException("Unparsable feature: " + a)
      }
    }
    fvbuf.toArray
  }


  def getFeatureAlphabetSingleton(fstr: String): (Alphabet, Alphabet) = {
    val ifile = new java.io.File(fstr)
    val fa = new StdAlphabet()
    val la = new StdAlphabet()
    val lines = scala.io.Source.fromFile(ifile).getLines
    lines foreach { l =>
      l.split(' ').toList match {
        case _ :: lab :: rest =>
          la.ofString(lab)
          FactorGraph.getLineInfo(rest, fa, false)
        case _ =>
      }
    }
    (fa, la)
  }

  def getFeatureAlphabetFactor(fstr: String): Alphabet = {
    val ifile = new java.io.File(fstr)
    val fa = new StdAlphabet()
    val lines = scala.io.Source.fromFile(ifile).getLines
    lines foreach { l =>
      val p = l.split(' ').toList
      val lab = p.head
      val rest = p.tail
      FactorGraph.getLineInfo(rest, fa, false)
    }
    fa
  }

  def gatherFactorGraph(fgSettings: FactorGraphSettings): FactorGraph = {
    val singletonStr = fgSettings.singletonFile
    val factorStr = fgSettings.factorFile
    val (sa, sla) = getFeatureAlphabetSingleton(singletonStr)
    sa.ensureFixed
    sla.ensureFixed
    val singletonExtractor = new GMSingletonExtractor(sa, sla, fgSettings.isSparse)
    val singletons = scala.io.Source.fromFile(new java.io.File(singletonStr)).getLines.toList map singletonExtractor.extractFeatures
    logger.info("Factor graph constructed with " + sa.getSize + " features for singleton factors")
    val cardinality = sla.getSize // # of labels
    factorStr match {
      case Some(factorStr) =>
        val fa = getFeatureAlphabetFactor(factorStr)
        val fla = new IdentityAlphabet(cardinality * cardinality, false, true)
        fa.ensureFixed
        fla.ensureFixed
        val factorExtractor = new GMFactorExtractor(fa, fla, cardinality, singletonExtractor.variableToSingletonFactors, fgSettings.factorSparse)
        logger.info("Factor graph constructed with " + fa.getSize + " features for pair-wise factors")
        val factors = scala.io.Source.fromFile(new java.io.File(factorStr)).getLines.toList map factorExtractor.extractFeatures
        new FactorGraph(factors.toVector, singletons.toVector, AlphabetSet(sa, fa, sla, fla))
      case None =>
        new FactorGraph(Vector(), singletons.toVector, AlphabetSet(sa, new IdentityAlphabet, sla, new IdentityAlphabet))
    }
  }

  def gatherFactorGraph(fgSettings: FactorGraphSettings, alphabets: AlphabetSet) = {
    val sla = alphabets.sla
    val sa = alphabets.sa
    val fla = alphabets.fla
    val fa = alphabets.fa
    val singletonStr = fgSettings.singletonTestFile.get

    val singletonExtractor = new GMSingletonExtractor(sa, sla, fgSettings.isSparse)
    val singletons = scala.io.Source.fromFile(new java.io.File(singletonStr)).getLines.toList map singletonExtractor.extractFeatures
    logger.info("Factor graph constructed with " + sa.getSize + " features for singleton factors")
    fgSettings.factorTestFile match {
      case Some(factorStr) =>
        val factorExtractor = new GMFactorExtractor(fa, fla, sla.getSize, singletonExtractor.variableToSingletonFactors, fgSettings.factorSparse)
        val factors = scala.io.Source.fromFile(new java.io.File(factorStr)).getLines.toList map factorExtractor.extractFeatures
        logger.info("Factor graph constructed with " + fa.getSize + " features for pair-wise factors")
        new FactorGraph(factors.toVector, singletons.toVector, alphabets)
      case None =>
        new FactorGraph(Vector(), singletons.toVector, alphabets)
    }
  }
}

class IdentityExtractor(alphabet: Alphabet) extends FeatureExtractor[SingletonFactor, MMLPFactor] {
  def getAlphabet = alphabet

  def getNumberOfFeatures = alphabet.getSize

  def extractFeatures(fm: SingletonFactor): MMLPFactor = fm.getInput
}

class GMFactorExtractor(factorAlphabet: Alphabet, factorLa: Alphabet, varOrder: Int, varToSingles: Map[String, SingletonFactor],
                        sparse: Boolean = false)
  extends FeatureExtractor[String, MultiFactor] {

  val sep = ' '
  val buildVecs = true
  val factorSize = 2

  val bases = Vector.tabulate(factorSize) { i => math.pow(varOrder, i).toInt }
  val numConfigs = math.pow(varOrder, factorSize).toInt

  def getAlphabet = factorAlphabet

  def getNumberOfFeatures = factorAlphabet.getSize

  def indexToAssignment(ind: Int): Array[Int] = {
    val ar = Array.fill(factorSize)(0)
    var i = factorSize - 1
    var nn = ind
    while (i >= 0) {
      val base = bases(i)
      ar(i) = nn / base
      nn -= (base * ar(i))
      i -= 1
    }
    ar
  }


  val indexAssignmentMap = Array.tabulate(numConfigs)(indexToAssignment)

  def extractFeatures(s: String): MultiFactor = {

    val p = s.split(sep).toList
    val lab = p.head
    val rest = p.tail

    val lb = lab.split('-').toList
    lb match {
      case v1 :: v2 :: Nil =>
        val fv = FactorGraph.getLineInfo(rest, factorAlphabet, true)
        val vec: Vec = if (sparse) SparseVec(factorAlphabet.getSize) else DenseVec.zeros(factorAlphabet.getSize)
        fv foreach { f =>
          if (f.fid >= 0) {
            val fv = factorAlphabet.getValue(f.fid, f.value).toFloat
            vec.update(f.fid, fv)
          }
        }
        val s1 = varToSingles(v1)
        val s2 = varToSingles(v2)
        val sgs = Array(s1, s2)
        val mf = new MultiFactor(indexAssignmentMap, varOrder, sgs, vec, (v1 + v2))
        s1.addParent(mf, 0)
        s2.addParent(mf, 1)
        mf
      case a => throw new RuntimeException("Invalid input: " + a)
    }
  }
}

class GMSingletonExtractor(singletonAlphabet: Alphabet, singletonLa: Alphabet, sparse: Boolean = false) extends FeatureExtractor[String, SingletonFactor] {

  val sep = ' '
  val buildVecs = true


  var variableToSingletonFactors: Map[String, SingletonFactor] = Map[String, SingletonFactor]()

  def getAlphabet = singletonAlphabet

  def getNumberOfFeatures = singletonAlphabet.getSize


  def extractFeatures(s: String): SingletonFactor = {
    val p = s.split(sep).toList
    val uniqueId = p.head
    val rest = p.tail

    val lb = uniqueId.split('-').toList // ensure not a factor
    lb match {
      case v1 :: Nil =>
        val label = rest.head
        val fstrs = rest.tail
        val fv = FactorGraph.getLineInfo(fstrs, singletonAlphabet, true)
        val vec: Vec = if (sparse) SparseVec(singletonAlphabet.getSize) else DenseVec.zeros(singletonAlphabet.getSize)
        fv foreach { f =>
          if (f.fid >= 0) {
            val fv = singletonAlphabet.getValue(f.fid, f.value).toFloat
            vec.update(f.fid, fv)
          }
        }
        val l_ind = math.max(singletonLa.ofString(label), 0)
        val lv = DenseVec.zeros(singletonLa.getSize)
        lv.update(l_ind, 1.0f) // one-hot encoding
      val sgf = vec match {
        case v: DenseVec => new StdMMLPFactor(-1, v, lv, Some(uniqueId))
        case v: SparseVec => new SparseMMLPFactor(-1, v, lv, Some(uniqueId))
      }
        val factor = new SingletonFactor(sgf, l_ind)
        variableToSingletonFactors += (v1 -> factor)
        factor

      case _ => throw new RuntimeException("Invalid input: " + s)
    }
  }

}
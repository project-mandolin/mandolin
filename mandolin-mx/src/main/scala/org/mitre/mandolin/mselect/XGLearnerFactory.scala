package org.mitre.mandolin.mselect


import org.mitre.mandolin.xg.{XGBoostEvaluator, XGModelSettings}
import org.mitre.mandolin.glp.{LType, InputLType, SparseInputLType, SoftMaxLType, GLPFactor}

class XGModelSpaceBuilder(ms: Option[ModelSpace]) extends ModelSpaceBuilder {

  def this(m: ModelSpace) = this(Some(m))

  def this() = this(None)

  // initialize with modelspace
  ms foreach { ms =>
    ms.catMPs foreach withMetaParam
    ms.realMPs foreach withMetaParam
    ms.intMPs foreach withMetaParam

  }

  // XXX - clean/refactor
  def build(appSettings: Option[XGModelSettings]): ModelSpace = {
    val budget = appSettings match {
      case Some(m) => m.numEpochs
      case None => -1
    }
    // Pull out important parameters to preserve here and pass into model space
    val appConfig = appSettings map { a => a.config.root.render() }
    new ModelSpace(reals.toVector, cats.toVector, ints.toVector, -1, -1, appConfig, budget)
  }
}


class XGModelInstance(appSettings: XGModelSettings, nfs: Int, modelId: Int, startFrom: Int)
  extends LearnerInstance[GLPFactor] {

  def train(trVecs: Vector[GLPFactor], tstVecs: Option[Vector[GLPFactor]]): Double = {
    tstVecs match {
      case Some(tstVecs) =>
        val evaluator = new XGBoostEvaluator(appSettings)
        val (finalMetric, _) = evaluator.evaluateTrainingSet(trVecs.toIterator, Some(tstVecs.toIterator))
        finalMetric.toDouble
      case None =>
        val evaluator = new XGBoostEvaluator(appSettings)
        // this will do x-validation if not supplied with tst data
        val (finalMetric, _) = evaluator.evaluateTrainingSet(trVecs.toIterator, None)
        finalMetric.toDouble
    }
  }
}

trait XGLearnerBuilderHelper {
  def setupSettings(config: ModelConfig): XGModelSettings = {
    val budget = config.budget // number of iterations, typically
    val cats: List[(String, Any)] = config.categoricalMetaParamSet.toList map { cm => (cm.getName, cm.getValue.s) }
    val reals: List[(String, Any)] = config.realMetaParamSet.toList map { cm => (cm.getName, cm.getValue.v) }
    val ints: List[(String, Any)] = config.intMetaParamSet.toList map { cm => (cm.getName, cm.getValue.v) }
    val setBudget: List[(String, Any)] = if (budget > 0) List(("mandolin.trainer.num-epochs", budget)) else Nil
    val allParams: Seq[(String, Any)] = (cats ++ reals ++ ints ++ setBudget) toSeq
    val completeParams = allParams
    val xgsets = config.serializedSettings match {
      case Some(s) => new XGModelSettings(s)
      case None => new XGModelSettings()
    }
    xgsets.withSets(completeParams)
  }
}

object XGModelInstance extends XGLearnerBuilderHelper {
  def apply(config: ModelConfig): XGModelInstance = {
    val settings = setupSettings(config)
    new XGModelInstance(settings, config.inDim, config.id, config.src)
  }
}


class XGModelEvaluator(trData: Vector[GLPFactor], tstData: Option[Vector[GLPFactor]]) extends ModelEvaluator with Serializable {

  def evaluate(c: ModelConfig, generation: Int): (Double, Long) = {
    val learner = XGModelInstance(c)
    val startTime = System.currentTimeMillis()
    val acc = learner.train(trData, tstData)
    val endTime = System.currentTimeMillis()
    (acc, endTime - startTime)
  }

  override def cancel(generation: Int): Unit = {}
}

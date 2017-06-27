package org.mitre.mandolin.mselect

import org.mitre.mandolin.mlp.MandolinMLPSettings
import scala.collection.JavaConversions._

trait ModelSelectionSettings extends MandolinMLPSettings {
  
  private def getIntPair(li: List[Integer]) = li match {case a :: b :: Nil => (a.toInt, b.toInt) case _ => throw new RuntimeException("Invalid integer range")}
  private def getIntTriple(li: List[Integer]) = 
    li match {case a :: b :: c :: Nil => (a.toInt, b.toInt, c.toInt) case _ => throw new RuntimeException("Invalid integer range")}
  private def getDoublePair(li: List[java.lang.Double]) = li match {case a :: b :: Nil => (a.toDouble, b.toDouble) case _ => throw new RuntimeException("Invalid integer range")}
  private def getDoubleTriple(li: List[java.lang.Double]) = 
    li match {case a :: b :: c :: Nil => (a.toDouble, b.toDouble, c.toDouble) case _ => throw new RuntimeException("Invalid integer range_by")}
  
  def buildModelSpaceFromConfig() = {
    val cobj = config.getConfig("mandolin.model-selection")
    
    val cats = cobj.getConfigList("categorical") map {c => 
      val key = c.getString("name")
      // these values should be strings
      val values = c.getStringList("values") 
      CategoricalMetaParameter(key, new CategoricalSet(values.toVector))
    }
    val reals = cobj.getConfigList("real") map {c =>
      val key = c.getString("name")
      try {
        val (l,u) = getDoublePair(c.getDoubleList("range").toList)      
        RealMetaParameter(key, new RealSet(l,u))      
      } catch {
        case _: Throwable =>
          val (l,u,s) = getDoubleTriple(c.getDoubleList("range_by").toList)      
          RealMetaParameter(key, new StepRealSet(l,u,s))
      }
    }
    val ints =  cobj.getConfigList("int") map {c =>
      val key = c.getString("name")
      try {
        val (l,u) = getIntPair(c.getIntList("range").toList) 
        IntegerMetaParameter(key, new IntSet(l,u))
      } catch {
        case _: Throwable =>
          val (l,u,s) = getIntTriple(c.getIntList("range_by").toList)      
          IntegerMetaParameter(key, new StepIntSet(l,u,s))
      }
    }
    val budget = if (this.useHyperband) this.numEpochs else -1
    new ModelSpace(reals.toVector, cats.toVector, ints.toVector, 0, 0, None, budget)
  }
  
  val modelSpace = buildModelSpaceFromConfig()
  val useHyperband = asBoolean("mandolin.model-selection.use-hyperband")
  val useCheckpointing = asBoolean("mandolin.model-selection.use-checkpoints")
  val hyperbandMixParam = asFloatOpt("mandolin.model-selection.hyper-acq-mix").getOrElse(1.0f) // coefficient to mix in Bayesian optimization
  val numWorkers = asInt("mandolin.model-selection.concurrent-evaluations")
  val threadsPerWorker = asInt("mandolin.model-selection.threads-per-worker")
  val workerBatchSize = asInt("mandolin.model-selection.worker-batch-size")
  val scoreSampleSize = asInt("mandolin.model-selection.score-sample-size")
  val updateFrequency = asInt("mandolin.model-selection.update-frequency")
  val totalEvals      = asInt("mandolin.model-selection.total-evals")  
  val acquisitionFunction = asStrOpt("mandolin.model-selection.acquisition-function") match {
    case Some("random") => new RandomAcquisition
    case Some("ucb-3") => new UpperConfidenceBound(0.3)
    case Some("ucb-7") => new UpperConfidenceBound(0.7)
    case Some("ucb") => new UpperConfidenceBound(0.3)
    case Some("pi") => new ProbabilityOfImprovement
    case Some("ei2") => new ExpectedImprovementVer2
    case _ => new ExpectedImprovement
  }
}
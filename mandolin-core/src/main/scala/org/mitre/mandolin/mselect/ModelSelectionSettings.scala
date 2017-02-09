package org.mitre.mandolin.mselect

import com.typesafe.config.{ConfigList, ConfigValueType}
import org.mitre.mandolin.config.LearnerSettings

import scala.collection.JavaConversions._

trait ModelSelectionSettings extends LearnerSettings {
  
  def buildModelSpaceFromConfig() = {
    val cobj = config.getConfig("mandolin.model-selection")
        
    val cats = cobj.getConfigList("categorical") map {c => 
      val key = c.getString("name")
      // these values should be strings
      val values = c.getStringList("values") 
      new CategoricalMetaParameter(key, new CategoricalSet(values.toVector))
    }
    val reals = cobj.getConfigList("real") map {c =>
      val key = c.getString("name")
      val (l,u) = c.getDoubleList("range").toList match {case a :: b :: Nil => (a.toDouble,b.toDouble) case _ => throw new RuntimeException("invalid range")}
      new RealMetaParameter(key, new RealSet(l,u))
      }
    val ints =  cobj.getConfigList("int") map {c =>
      val key = c.getString("name")
      val (l,u) = c.getIntList("range").toList match {case a :: b :: Nil => (a,b) case _ => throw new RuntimeException("invalid range")}
      new IntegerMetaParameter(key, new IntSet(l,u))
      }
    new ModelSpace(reals.toVector, cats.toVector, ints.toVector)
  }
  
  val modelSpace = buildModelSpaceFromConfig()
  val numWorkers = asInt("mandolin.model-selection.num-workers")
  val threadsPerWorker = asInt("mandolin.model-selection.threads-per-worker")
  val workerBatchSize = asInt("mandolin.model-selection.worker-batch-size")
  val scoreSampleSize = asInt("mandolin.model-selection.score-sample-size")
  val updateFrequency = asInt("mandolin.model-selection.update-frequency")
  val totalEvals      = asInt("mandolin.model-selection.total-evals")
  
}
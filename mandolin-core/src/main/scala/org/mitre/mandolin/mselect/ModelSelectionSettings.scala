package org.mitre.mandolin.mselect

import com.typesafe.config.{ConfigList, ConfigValueType}
import org.mitre.mandolin.config.LearnerSettings

import scala.collection.JavaConversions._

trait ModelSelectionSettings extends LearnerSettings {
  
  private def getIntPair(li: List[Integer]) = li match {case a :: b :: Nil => (a.toInt, b.toInt) case _ => throw new RuntimeException("Invalid integer range")}
  private def getDoublePair(li: List[java.lang.Double]) = li match {case a :: b :: Nil => (a.toDouble, b.toDouble) case _ => throw new RuntimeException("Invalid integer range")}
  
  def buildModelSpaceFromConfig() = {
    val cobj = config.getConfig("mandolin.model-selection")
        
    val cats = cobj.getConfigList("categorical") map {c => 
      val key = c.getString("name")
      // these values should be strings
      val values = c.getStringList("values") 
      CategoricalMetaParameter(key, CategoricalSet(values.toVector))
    }
    val reals = cobj.getConfigList("real") map {c =>
      val key = c.getString("name")
      val (l,u) = getDoublePair(c.getDoubleList("range").toList)
      RealMetaParameter(key, RealSet(l,u))
      }
    val ints =  cobj.getConfigList("int") map {c =>
      val key = c.getString("name")
      val (l,u) = getIntPair(c.getIntList("range").toList) 
      IntegerMetaParameter(key, IntSet(l,u))
      }
    // this defines a vector of "topology meta parameters"
    // each topologyMetaParameter defines a space of topologies for a fixed number of layers
    // a topologySpaceMetaParameter is then a set/vector of these
    // this allows for the space to be tailored/bounded in a reasonable way for different
    val layers = try {
      val vec = cobj.getConfigList("layers") map {l =>    
      val key = l.getString("name") // just the layer name
      val topo = l.getConfigList("topology")
      val topoLayers = topo map { t =>
        val lt = t.getStringList("ltype")
        val (lDim,uDim) = getIntPair(t.getIntList("dim").toList)
        val (ll1,ul1) = getDoublePair(t.getDoubleList("l1-pen").toList) 
        val (ll2,ul2) = getDoublePair(t.getDoubleList("l2-pen").toList)
        new LayerMetaParameter("layer",
            TupleSet4 (
              CategoricalMetaParameter("ltype", CategoricalSet(lt.toVector)),
              IntegerMetaParameter("dim", IntSet(lDim, uDim)), 
              RealMetaParameter("l1pen", RealSet(ll1, ul1)),
              RealMetaParameter("l2pen", RealSet(ll2, ul2)) ))
        } 
      new TopologyMetaParameter(key, topoLayers.toVector)
      }
      vec.toVector} catch {case _: Throwable => Vector()}
    val ll = ListSet(layers.toVector)
    val sp = if (ll.size > 0) Some(new TopologySpaceMetaParameter("topoSpace", ll)) else None
    new ModelSpace(reals.toVector, cats.toVector, ints.toVector, sp)
  }
  
  val modelSpace = buildModelSpaceFromConfig()
  val numWorkers = asInt("mandolin.model-selection.num-workers")
  val threadsPerWorker = asInt("mandolin.model-selection.threads-per-worker")
  val workerBatchSize = asInt("mandolin.model-selection.worker-batch-size")
  val scoreSampleSize = asInt("mandolin.model-selection.score-sample-size")
  val updateFrequency = asInt("mandolin.model-selection.update-frequency")
  val totalEvals      = asInt("mandolin.model-selection.total-evals")
  
}
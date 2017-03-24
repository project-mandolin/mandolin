package org.mitre.mandolin.mselect

import org.mitre.mandolin.glp.{LType, InputLType, SoftMaxLType, SparseInputLType, LinearLType }
import com.typesafe.config.{ConfigList, ConfigValueType}
import org.mitre.mandolin.config.GeneralLearnerSettings
import org.mitre.mandolin.glp.GLPModelSettings
import net.ceedubs.ficus.Ficus._

import scala.collection.JavaConversions._

trait ModelSelectionSettings extends GLPModelSettings {
  
  private def getIntPair(li: List[Integer]) = li match {case a :: b :: Nil => (a.toInt, b.toInt) case _ => throw new RuntimeException("Invalid integer range")}
  private def getIntTriple(li: List[Integer]) = 
    li match {case a :: b :: c :: Nil => (a.toInt, b.toInt, c.toInt) case _ => throw new RuntimeException("Invalid integer range")}
  private def getDoublePair(li: List[java.lang.Double]) = li match {case a :: b :: Nil => (a.toDouble, b.toDouble) case _ => throw new RuntimeException("Invalid integer range")}
  private def getDoubleTriple(li: List[java.lang.Double]) = 
    li match {case a :: b :: c :: Nil => (a.toDouble, b.toDouble, c.toDouble) case _ => throw new RuntimeException("Invalid integer range_by")}
  
  def buildModelSpaceFromConfig() = {
    val cobj = config.getConfig("mandolin.model-selection")
    
    val (inLType, outLType) = {
      //val (is, os) = config.getConfigList("mandolin.trainer.specification").toList match {
      val li = try {
        config.as[List[Map[String,String]]]("mandolin.trainer.specification")
      } catch {case _: Throwable => this.mapSpecToList(config.as[Map[String, Map[String, String]]]("mandolin.trainer.specification"))}
      val (is, os) = 
      li match {
        case a :: rest => (a, rest.reverse.head)
        case _ => throw new RuntimeException("Invalid mandolin.trainer.specification")
        }
      val inLType = is("ltype") match {
        case "Input" => LType(InputLType)
        case "InputSparse" => LType(SparseInputLType)
        case a => throw new RuntimeException("Invalid LType " + a)
      }
      val outLType = os("ltype") match {
        case "SoftMax" => LType(SoftMaxLType)
        case "Linear" => LType(LinearLType)
        case a => throw new RuntimeException("Invalid LType " + a)
      }
      (inLType, outLType)
    }
        
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
              CategoricalMetaParameter("ltype", new CategoricalSet(lt.toVector)),
              IntegerMetaParameter("dim", new IntSet(lDim, uDim)), 
              RealMetaParameter("l1pen", new RealSet(ll1, ul1)),
              RealMetaParameter("l2pen", new RealSet(ll2, ul2)) ))
        } 
      new TopologyMetaParameter(key, topoLayers.toVector)
      }
      vec.toVector} catch {case _: Throwable => Vector()}
    val ll = ListSet(layers.toVector)
    val sp = if (ll.size > 0) Some(new TopologySpaceMetaParameter("topoSpace", ll)) else None
    new ModelSpace(reals.toVector, cats.toVector, ints.toVector, sp, inLType, outLType, 0, 0, None)
  }
  
  val modelSpace = buildModelSpaceFromConfig()
  val numWorkers = asInt("mandolin.model-selection.num-workers")
  val threadsPerWorker = asInt("mandolin.model-selection.threads-per-worker")
  val workerBatchSize = asInt("mandolin.model-selection.worker-batch-size")
  val scoreSampleSize = asInt("mandolin.model-selection.score-sample-size")
  val updateFrequency = asInt("mandolin.model-selection.update-frequency")
  val totalEvals      = asInt("mandolin.model-selection.total-evals")  
  val acquisitionFunction = asStrOpt("mandolin.model-selection.acquisition-function") match {
    case Some("random") => new RandomAcquisitionFunction
    case None => new BayesianNNAcquisitionFunction(modelSpace)
  }
}
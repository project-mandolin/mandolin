package org.mitre.mandolin.mx

import org.mitre.mandolin.optimize.{ Updater, LossGradient, TrainingUnitEvaluator }
import org.mitre.mandolin.mlp.MMLPFactor
import ml.dmlc.mxnet.{ EpochEndCallback, Uniform, Initializer, Xavier, Model, Shape, FeedForward, 
  Symbol, Context, Optimizer, NDArray, DataIter, DataBatch, Accuracy }

import ml.dmlc.mxnet.io.{ NDArrayIter }
import ml.dmlc.mxnet.optimizer.SGD
import ml.dmlc.mxnet.Callback.Speedometer
import collection.mutable.ArrayBuffer
import org.slf4j.LoggerFactory

class MxNetLossGradient(loss: Double) extends LossGradient[MxNetLossGradient](loss) {
  def add(other: MxNetLossGradient) = new MxNetLossGradient(this.loss + other.loss)
  def asArray = throw new RuntimeException("Loss gradient not convertable to array")
  
}

class MxNetFactor(val data: NDArray, val label: NDArray) extends Serializable 

/**
 * This iterator creates a MxNet DataIter from Iterator[DataBatch]
 * THis allows the API to remain separate from MxNet but should allow mxNet DataIter objects to be
 * passed in directly to evaluateTrainingMiniBatch since DataIter extends Iterator[DataBatch]
 */
class GenMxIter(it: Iterator[DataBatch], _batchSize: Int, _shape: Shape) extends GenIter(_shape, _batchSize, "data", "softmax_label") {
  
  override def next() : DataBatch = {
    if (!hasNext) {
      throw new NoSuchElementException("No more data")
    }
    index += 1
    if (index >= 0 && index < cache.size) {
      cache(index)
    } else {
      val db = it.next()
      cache += db
      db
    }
  }
  
  override def hasNext() : Boolean = it.hasNext || (index < cache.size - 1 && cache.size > 0)
}

class MxModelCheckPoint(prefix: String, freq: Int = 1) extends EpochEndCallback {
  def invoke(epoch: Int, symbol: Symbol, argParams: Map[String, NDArray], auxParams: Map[String, NDArray]) : Unit = {
    if ((epoch+1) % freq == 0)
      Model.saveCheckpoint(prefix, epoch, symbol, argParams, auxParams)
  }
}

class MxNetEvaluator(val net: Symbol, val ctx: Array[Context], shape: Shape, batchSz: Int, init: Initializer,
    checkPointPrefix: Option[String] = None, checkPointFreq: Int = 1)
extends TrainingUnitEvaluator[DataBatch, MxNetWeights, MxNetLossGradient, MxNetOptimizer] {
  
  val logger = LoggerFactory.getLogger(getClass)
  val checkPointer = 
    if (checkPointFreq > 0) checkPointPrefix match {case Some(p) => new MxModelCheckPoint(p, checkPointFreq) case None => null}
    else null
    
  def evaluateTrainingUnit(unit: DataBatch, weights: MxNetWeights, u: MxNetOptimizer) : MxNetLossGradient = 
    throw new RuntimeException("Closed evaluator MxNetEvaluator does not implement singleton point evaluations")   
  
  def copy() = throw new RuntimeException("MxNetEvaluator should/can not be copied")
  
  def evaluateTrainingMiniBatch(tr: DataIter, tst: DataIter, weights: MxNetWeights, 
      u: MxNetOptimizer, epochCnt: Int = 0, startFrom: Int = -1) : MxNetLossGradient = {    
    val metric = new Accuracy()
    if ((checkPointPrefix.isDefined) && (startFrom > 0)) { // in this case we're resuming training from a saved checkpoint
      val epochSize = math.ceil(tr.size.toDouble / batchSz)
      val (sym, args, auxs) = Model.loadCheckpoint(checkPointPrefix.get, startFrom)
      logger.info("Loading model " + checkPointPrefix.get)
      val ff = new FeedForward(net, ctx, optimizer = u.optimizer, 
          initializer = init, numEpoch = epochCnt, batchSize = batchSz, argParams = args, auxParams = auxs,
          beginEpoch = startFrom, allowExtraParams = true)
      ff.fit(trainData = tr, evalData = tst, evalMetric = metric, kvStoreType = "local_update_cpu", epochEndCallback = checkPointer, 
          batchEndCallback = new Speedometer(batchSz, 50))
      checkPointPrefix foreach {p => Model.saveCheckpoint(p, epochCnt, sym, ff.getArgParams, ff.getAuxParams)}
    } else {

      val ff = new FeedForward(net, ctx, optimizer = u.optimizer, 
        initializer = init, numEpoch = epochCnt, batchSize = batchSz, argParams = null, auxParams = null)
      ff.fit(trainData = tr, evalData = tst, evalMetric = metric, kvStoreType = "local_update_cpu", epochEndCallback = checkPointer, 
          batchEndCallback = new Speedometer(batchSz, 50))
      weights.setArgParams(ff.getArgParams)
      weights.setAuxParams(ff.getAuxParams)
      checkPointPrefix foreach {p => Model.saveCheckpoint(p, epochCnt, net, ff.getArgParams, ff.getAuxParams)}
    }        
               
    new MxNetLossGradient(metric.get._2(0).toDouble)
  }
}



class MxNetGlpEvaluator(val net: Symbol, val ctx: Array[Context], idim: Int)
extends TrainingUnitEvaluator[MMLPFactor, MxNetWeights, MxNetLossGradient, MxNetOptimizer] {
  
  val batchSize = 64
  // indicates that only minibatch training is supported and gradient updates are handled by evaluator directly
  val isClosed = true
  
  private def factorsToIterator(units: Iterator[MMLPFactor]) : MMLPFactorIter = {
    new MMLPFactorIter(units, Shape(idim), batchSize)
  }
  
  def evaluateTrainingMiniBatch(units: Iterator[MMLPFactor], weights: MxNetWeights, u: MxNetOptimizer, epochCnt: Int = 0) : MxNetLossGradient = {
    // make a single epoch/pass over the data
    val args = weights.argParams.getOrElse(null) 
    val auxs = weights.auxParams.getOrElse(null)
    val ff = new FeedForward(net, ctx, optimizer = u.optimizer, numEpoch = epochCnt+1, 
        argParams = args, auxParams = auxs, beginEpoch=epochCnt)  
    val trIter = factorsToIterator(units)
    val tstIter = factorsToIterator(units) // separate iterators here
    val metric = new Accuracy()
    ff.fit(trIter, tstIter, metric)
    trIter.dispose() // clean out the cache for this iterator
    tstIter.dispose()
    new MxNetLossGradient(metric.get._2(0).toDouble)    
  }
  
  def evaluateTrainingUnit(unit: MMLPFactor, weights: MxNetWeights, u: MxNetOptimizer) : MxNetLossGradient =
    throw new RuntimeException("Closed evaluator MxNetEvaluator does not implement singleton point evaluations")   
  
  def copy() = throw new RuntimeException("MxNetEvaluator should/can not be copied")
}

class MxNetOptimizer(val optimizer: ml.dmlc.mxnet.Optimizer) extends Updater[MxNetWeights, MxNetLossGradient, MxNetOptimizer] {
  def asArray: Array[Float] = throw new RuntimeException("Unimplemented")
  def compose(u: org.mitre.mandolin.mx.MxNetOptimizer): org.mitre.mandolin.mx.MxNetOptimizer = {
    this
  }
  def compress(): org.mitre.mandolin.mx.MxNetOptimizer = this
  def copy(): org.mitre.mandolin.mx.MxNetOptimizer = throw new RuntimeException("MxNetOptimizer should/can not be copied")
  def decompress(): org.mitre.mandolin.mx.MxNetOptimizer = this
  def resetLearningRates(v: Float): Unit = {}
  def updateFromArray(a: Array[Float]): Unit = {}
  def updateWeights(g: org.mitre.mandolin.mx.MxNetLossGradient,w: org.mitre.mandolin.mx.MxNetWeights): Unit = {}
}
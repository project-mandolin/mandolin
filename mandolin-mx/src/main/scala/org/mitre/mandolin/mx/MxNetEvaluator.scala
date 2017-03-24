package org.mitre.mandolin.mx

import org.mitre.mandolin.optimize.{ Updater, LossGradient, TrainingUnitEvaluator }
import org.mitre.mandolin.glp.GLPFactor
import ml.dmlc.mxnet.{ EpochEndCallback, Xavier, Model, Shape, FeedForward, Symbol, Context, Optimizer, NDArray, DataIter, DataBatch, Accuracy }
import ml.dmlc.mxnet.io.{ NDArrayIter }
import ml.dmlc.mxnet.optimizer.SGD
import ml.dmlc.mxnet.Callback.Speedometer
import collection.mutable.ArrayBuffer

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

class MxNetEvaluator(val net: Symbol, val ctx: Array[Context], shape: Shape, batchSz: Int, 
    checkPointPrefix: Option[String] = None, checkPointFreq: Int = 1)
extends TrainingUnitEvaluator[DataBatch, MxNetWeights, MxNetLossGradient, MxNetOptimizer] {
  
  val checkPointer = checkPointPrefix match {case Some(p) => new MxModelCheckPoint(p, checkPointFreq) case None => null}
  def evaluateTrainingUnit(unit: DataBatch, weights: MxNetWeights, u: MxNetOptimizer) : MxNetLossGradient = 
    throw new RuntimeException("Closed evaluator MxNetEvaluator does not implement singleton point evaluations")   
  
  def copy() = throw new RuntimeException("MxNetEvaluator should/can not be copied")
  
  def evaluateTrainingMiniBatch(tr: DataIter, tst: DataIter, weights: MxNetWeights, u: MxNetOptimizer, epochCnt: Int = 0) : MxNetLossGradient = {    
    var acc = 0.0
    val metric = new Accuracy()
    val ff = new FeedForward(net, ctx, optimizer = u.optimizer, 
        initializer = new Xavier(rndType = "gaussian", factorType = "in", magnitude = 2.0f), 
            numEpoch = epochCnt, batchSize = batchSz, argParams = null, auxParams = null)
    ff.fit(trainData = tr, evalData = tst, evalMetric = metric, kvStoreType = "local_update_cpu", epochEndCallback = checkPointer, 
          batchEndCallback = new Speedometer(batchSz, 50))
    weights.setArgParams(ff.getArgParams)
    weights.setAuxParams(ff.getAuxParams)
    acc = metric.get._2.toDouble        
    new MxNetLossGradient(acc)
  }
}



class MxNetGlpEvaluator(val net: Symbol, val ctx: Array[Context], idim: Int)
extends TrainingUnitEvaluator[GLPFactor, MxNetWeights, MxNetLossGradient, MxNetOptimizer] {
  
  val batchSize = 64
  // indicates that only minibatch training is supported and gradient updates are handled by evaluator directly
  val isClosed = true
  
  private def factorsToIterator(units: Iterator[GLPFactor]) : GLPFactorIter = {    
    new GLPFactorIter(units, Shape(idim), batchSize)
  }
  
  def evaluateTrainingMiniBatch(units: Iterator[GLPFactor], weights: MxNetWeights, u: MxNetOptimizer, epochCnt: Int = 0) : MxNetLossGradient = {
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
    new MxNetLossGradient(metric.get._2.toDouble)    
  }
  
  def evaluateTrainingUnit(unit: GLPFactor, weights: MxNetWeights, u: MxNetOptimizer) : MxNetLossGradient = 
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
package org.mitre.mandolin.mx

import org.mitre.mandolin.mlp.MMLPFactor
import ml.dmlc.mxnet.{ DataIter, DataBatch, Shape, NDArray }
import scala.collection.immutable.ListMap
import scala.collection.mutable.ArrayBuffer

abstract class GenIter(_shape: Shape, _batchSize: Int, dataName: String, labelName: String) extends DataIter {
  protected val cache: ArrayBuffer[DataBatch] = ArrayBuffer.empty[DataBatch]
  protected var index: Int = -1
  protected val dataShape : Shape = _shape

  def dispose(): Unit = {
    cache.foreach(_.dispose())
  }
  
  /**
   * reset the iterator
   */
  override def reset(): Unit = {
    index = -1
  }
  /** 
   * get data of current batch
   * @return the data of current batch
   */
  override def getData(): IndexedSeq[NDArray] = {
    if (index >= 0 && index < cache.size) {
      cache(index).data
    } else {
      null
    }
  }

  /**
   * Get label of current batch
   * @return the label of current batch
   */
  override def getLabel(): IndexedSeq[NDArray] = {
    if (index >= 0 && index < cache.size) {
      cache(index).label
    } else {
      null
    }
  }

  /**
   * Get the index of current batch
   * @return the index of current batch
   */
  override def getIndex(): IndexedSeq[Long] = {
    if (index >= 0 && index < cache.size) {
      cache(index).index
    } else {
      null
    }
  }

  // The name and shape of label provided by this iterator
  
  override def provideLabel: ListMap[String, Shape] = {
    ListMap(labelName -> Shape(_batchSize))
  }

  // The name and shape of data provided by this iterator
  override def provideData: ListMap[String, Shape] = {
    ListMap(dataName -> dataShape)
  }
  
  /**
   * Get the number of padding examples
   * in current batch
   * @return number of padding examples in current batch
   */
  override def getPad(): Int = 0

  override def batchSize: Int = _batchSize

}

class MMLPFactorIter(
                     val points: Iterator[MMLPFactor],
                     dimension: Shape,
                     _batchSize: Int,
                     _dataName: String = "data",
                     _labelName: String = "softmax_label") extends GenIter(dimension, _batchSize, _dataName, _labelName) {
  
  override protected val dataShape : Shape = Shape(_batchSize) ++ dimension
  
  @throws(classOf[NoSuchElementException])
  override def next(): DataBatch = {
    if (!hasNext) {
      throw new NoSuchElementException("No more data")
    }
    index += 1
    if (index >= 0 && index < cache.size) {
      cache(index)
    } else {
      val dataBuilder = NDArray.empty(dataShape)
      val labelBuilder = NDArray.empty(_batchSize)
      var instNum = 0
      while (instNum < batchSize && points.hasNext) {
        val point = points.next()
        val features = point.getInput.asArray
        require(features.length == dimension.product,
          s"Dimension mismatch: ${features.length} != $dimension")
        dataBuilder.slice(instNum).set(features)
        labelBuilder.slice(instNum).set(Array(point.getOneHot.toFloat)) // TODO - is this correct way to include classification output???
        instNum += 1
      }
      val pad = batchSize - instNum
      val dataBatch = new LongLivingDataBatch(
        IndexedSeq(dataBuilder), IndexedSeq(labelBuilder), null, pad)
      cache += dataBatch
      dataBatch
    }
  }

  
  override def hasNext: Boolean = {
    points.hasNext || (index < cache.size - 1 && cache.size > 0)
  }
}

/**
 */
class LongLivingDataBatch(
  override val data: IndexedSeq[NDArray],
  override val label: IndexedSeq[NDArray],
  override val index: IndexedSeq[Long],
  override val pad: Int) extends DataBatch(data, label, index, pad) {
  override def dispose(): Unit = {}
  def disposeForce(): Unit = super.dispose()
}
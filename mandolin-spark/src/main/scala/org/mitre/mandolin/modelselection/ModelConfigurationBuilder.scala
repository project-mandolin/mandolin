package org.mitre.mandolin.modelselection

import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.tuning.ParamGridBuilder

import scala.collection.mutable.ArrayBuffer

/**
  * Created by jkraunelis on 10/20/16.
  */
class ModelConfigurationBuilder extends ParamGridBuilder with Iterator[ParamMap] {

  protected val paramArray = ArrayBuffer.empty[(Param[_], Array[_], Int)]
  protected var cardinality = 0

  def addGrid[T](param: Param[T], values: Array[T]): this.type = {
    paramArray += ((param, values, 0))
    cardinality = paramArray.map(x => x._2.size).reduce((a, b) => a * b)
    this
  }

  def getUpdatedIndices(data: Array[(Int, Int)]) : Array[Int] = {
    var incNext = true
    data.map { case (size, i) => {
      val j = if (incNext) {
        if ( i+1 == size) {
          0
        } else {
          incNext = false
          i+1
        }
      } else {
        i
      }
      j
    }}
  }

  override def hasNext: Boolean = cardinality > 0

  override def next(): ParamMap = {
    cardinality -= 1
    val m = new ParamMap()
    paramArray.foreach { case (param, values, i) => m.put(param.asInstanceOf[Param[Any]], values(i)) }
    val newIndices = getUpdatedIndices(paramArray.map(x => (x._2.size, x._3)).toArray)
    val newArray = paramArray.zip(newIndices).map { case (pTuple, newIndex) => (pTuple._1, pTuple._2, newIndex) }
    paramArray.clear()
    paramArray ++= newArray
    m
  }
}

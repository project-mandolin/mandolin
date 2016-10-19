package org.mitre.mandolin.modelselection

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import scala.annotation.varargs
import scala.collection.mutable

import org.apache.spark.ml.param._

import scala.collection.mutable.ArrayBuffer

/**
  * Builder for a param grid used in grid search-based model selection.
  */

class LazyParamGridBuilder extends Iterator[ParamMap] {
  protected val paramArray = ArrayBuffer.empty[(Param[_], Array[_], Int)]
  protected var cardinality = 0
  /**
    * Adds a param with multiple values (overwrites if the input param exists).
    */
  def addGrid[T](param: Param[T], values: Array[T]): this.type = {
    paramArray += ((param, values, 0))
    cardinality = paramArray.map( x => x._2.size ).reduce( (a, b) => a*b)
    this
  }

  // specialized versions of addGrid for Java.

  /**
    * Adds a double param with multiple values.
    */
  def addGrid(param: DoubleParam, values: Array[Double]): this.type = {
    addGrid[Double](param, values)
  }

  /**
    * Adds an int param with multiple values.
    */
  def addGrid(param: IntParam, values: Array[Int]): this.type = {
    addGrid[Int](param, values)
  }

  /**
    * Adds a float param with multiple values.
    */
  def addGrid(param: FloatParam, values: Array[Float]): this.type = {
    addGrid[Float](param, values)
  }

  /**
    * Adds a long param with multiple values.
    */
  def addGrid(param: LongParam, values: Array[Long]): this.type = {
    addGrid[Long](param, values)
  }

  /**
    * Adds a boolean param with true and false.
    */
  def addGrid(param: BooleanParam): this.type = {
    addGrid[Boolean](param, Array(true, false))
  }

  def updateIndices(data: Array[(Int, Int)]) : Array[Int] = {
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
    paramArray.foreach{ case (param, values, i) => m.put(param, values(i))}
    val newIndices = updateIndices( paramArray.map( x => (x._2.size, x._3) ).toArray )
    val newArray = paramArray.zip(newIndices).map{ case (pTuple, newIndex) => (pTuple._1, pTuple._2, newIndex) }
    paramArray.clear()
    paramArray ++= newArray
    m
  }
}
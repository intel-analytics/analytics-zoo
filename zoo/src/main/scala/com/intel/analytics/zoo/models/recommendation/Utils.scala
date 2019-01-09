/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.models.recommendation

import com.intel.analytics.bigdl.dataset.{Sample, TensorSample}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.sql.functions.max
import org.apache.spark.sql.{DataFrame, Row}

import scala.util.Random

object Utils {

  def getNegativeSamples(indexed: DataFrame): DataFrame = {
    val schema = indexed.schema
    require(schema.fieldNames.contains("userId"), s"Column userId should exist")
    require(schema.fieldNames.contains("itemId"), s"Column itemId should exist")
    require(schema.fieldNames.contains("label"), s"Column label should exist")

    val indexedDF = indexed.select("userId", "itemId", "label")
    val minMaxRow = indexedDF.agg(max("userId"), max("itemId")).collect()(0)
    val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))
    val sampleDict = indexedDF.rdd.map(row => row(0) + "," + row(1)).collect().toSet

    val dfCount = indexedDF.count.toInt

    import indexed.sqlContext.implicits._

    @transient lazy val ran = new Random(System.nanoTime())

    val negative = indexedDF.rdd
      .map(x => {
        val uid = x.getAs[Int](0)
        val iid = Math.max(ran.nextInt(itemCount), 1)
        (uid, iid)
      })
      .filter(x => !sampleDict.contains(x._1 + "," + x._2)).distinct()
      .map(x => (x._1, x._2, 1))
      .toDF("userId", "itemId", "label")

    negative
  }

  def buckBucket(bucketSize: Int): (String, String) => Int = {
    val func = (col1: String, col2: String) =>
      (Math.abs((col1 + "_" + col2).hashCode()) % bucketSize + 0)
    func
  }

  // save 0 for uncovered ones
  def categoricalFromVocabList(vocabList: Array[String]): (String) => Int = {
    val func = (sth: String) => {
      val default: Int = 0
      val start: Int = 1
      if (vocabList.contains(sth)) vocabList.indexOf(sth) + start
      else default
    }
    func
  }

  def row2Sample(r: Row, columnInfo: ColumnFeatureInfo, modelType: String): Sample[Float] = {

    val wideTensor: Tensor[Float] = getWideTensor(r, columnInfo)
    val deepTensor: Tensor[Float] = getDeepTensor(r, columnInfo)
    val l = r.getAs[Int](columnInfo.label)

    val label = Tensor[Float](T(l))
    label.resize(1, 1)

    modelType match {
      case "wide_n_deep" =>
        TensorSample[Float](Array(wideTensor, deepTensor), Array(label))
      case "wide" =>
        TensorSample[Float](Array(wideTensor), Array(label))
      case "deep" =>
        TensorSample[Float](Array(deepTensor), Array(label))
      case _ =>
        throw new IllegalArgumentException("unknown type")
    }
  }

  // setup wide tensor
  def getWideTensor(r: Row, columnInfo: ColumnFeatureInfo): Tensor[Float] = {
    val wideColumns = columnInfo.wideBaseCols ++ columnInfo.wideCrossCols
    val wideDims = columnInfo.wideBaseDims ++ columnInfo.wideCrossDims
    val wideLength = wideColumns.length
    var acc = 0
    val indices: Array[Int] = (0 to wideLength - 1).map(i => {
      val index = r.getAs[Int](wideColumns(i))
      if (i == 0) index
      else {
        acc = acc + wideDims(i - 1)
        acc + index
      }
    }).toArray
    val values = indices.map(_ + 1.0f)
    val shape = Array(wideDims.sum)

    Tensor.sparse(Array(indices), values, shape)
  }

  // setup deep tensor
  def getDeepTensor(r: Row, columnInfo: ColumnFeatureInfo): Tensor[Float] = {
    val deepColumns1 = columnInfo.indicatorCols
    val deepColumns2 = columnInfo.embedCols ++ columnInfo.continuousCols
    val deepLength = columnInfo.indicatorDims.sum + deepColumns2.length
    val deepTensor = Tensor[Float](deepLength).fill(0)

    // setup indicators
    var acc = 0
    (0 to deepColumns1.length - 1).map {
      i => {
        val indexArray = r.getAs[Seq[Int]](columnInfo.indicatorCols(i))
        if (i == 0) {
          indexArray.foreach(index => deepTensor.setValue(index + 1, 1))
        } else {
          acc = acc + columnInfo.indicatorDims(i - 1)
          indexArray.foreach(index => deepTensor.setValue(acc + index + 1, 1))
        }
      }
    }

    // setup embedding and continuous
    (0 to deepColumns2.length - 1).map {
      i =>
        deepTensor.setValue(i + 1 + columnInfo.indicatorDims.sum,
          r.getAs[AnyVal](deepColumns2(i)) match {
            case i: Int => i.toFloat
            case l: Long => l.toFloat
            case f: Float => f
            case d: Double => d.toFloat
            case _ => throw new IllegalArgumentException("Please check the input data type of embedding and continuous cols.")
          })
    }
    deepTensor
  }
}

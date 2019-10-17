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

package com.intel.analytics.zoo.models.anomalydetection

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}


class AnomalyDetectorSpec extends ZooSpecHelper {

  var sqlContext: SQLContext = _
  var sc: SparkContext = _

  override def doBefore(): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setMaster("local[1]").setAppName("AnomalyTest")
    sc = NNContext.initNNContext(conf)
    sqlContext = SQLContext.getOrCreate(sc)
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "AnomalyDetector[Float] forward and backward " should "work properly" in {
    val model = AnomalyDetector[Float](Shape(10, 2), Array(10, 5), Array(0.2f, 0.2f))
    model.summary()
    val data: Seq[Tensor[Float]] = (1 to 50).map(i => {
      Tensor[Float](10, 10, 2).randn(0.0f, 0.1f)
    })
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }
  }

  "AnomalyDetector[Double] forward and backward" should "work properly" in {
    val model = AnomalyDetector[Double](Shape(10, 2), Array(10, 5), Array(0.2f, 0.2f))
    model.summary()
    val data: Seq[Tensor[Double]] = (1 to 50).map(i => {
      Tensor[Double](10, 10, 2).randn(0.0, 0.1)
    })
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }
  }

  "AnomalyDetector unroll " should "work properly" in {
    val data: RDD[Array[Float]] = sc.parallelize(5 to 104)
      .map(x => Array(x.toFloat))

    val unrolled = AnomalyDetector.unroll[Float](data, unrollLength = 3, predictStep = 2)

    val indicies: Array[Long] = unrolled.map(x => x.index).collect()
    val labels: Array[Float] = unrolled.map(x => x.label).collect()
    assert(unrolled.count() == 96)
    assert(indicies.max == 95 && indicies.min == 0)
    assert(labels.max == 104 && labels.min == 9)
  }

  "AnomalyDetector unroll multiple features " should "work properly" in {
    val data: RDD[Array[Float]] = sc.parallelize(5 to 104)
      .map(x => Array(x.toFloat))

    val unrolled = TimeSeriesPredictor.unroll[Float](data, encoderLength = 3, decoderLength=2)

      unrolled.take(10).foreach(println)
  }


}

class AnomalyDetectorSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {

    val model = AnomalyDetector[Float](Shape(10, 2), Array(10, 5), Array(0.2f, 0.2f))
    val data = Tensor[Float](10, 10, 2).randn(0.0, 0.1)
    ZooSpecHelper.testZooModelLoadSave(model, data, AnomalyDetector.loadModel[Float])
  }
}

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

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{RandomGenerator, Shape, T}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.anomalydetection.AnomalyDetector
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.apache.logging.log4j.{Level, LogManager}
import org.apache.logging.log4j.core.config.Configurator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}


class SessionRecommenderSpec extends ZooSpecHelper {

  var sqlContext: SQLContext = _
  var sc: SparkContext = _

  override def doBefore(): Unit = {
    Configurator.setLevel("org", Level.ERROR)
    val conf = new SparkConf().setMaster("local[1]").setAppName("NCFTest")
    sc = NNContext.initNNContext(conf)
    sqlContext = SQLContext.getOrCreate(sc)
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "SessionRecommender without history forward and backward" should "work properly" in {

    val itemCount = 100
    val sessionLength = 10
    val model = SessionRecommender[Float](itemCount, sessionLength = sessionLength)
    val ran = RandomGenerator.RNG
    val data = (1 to 100).map { x =>
      val items: Seq[Float] = for (i <- 1 to sessionLength) yield
        ran.uniform(1, itemCount).toInt.toFloat
      Tensor(items.toArray, Array(sessionLength)).resize(1, sessionLength)
    }
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }
  }

  "SessionRecommender with history forward and backward" should "work properly" in {
    val itemCount = 100
    val sessionLength = 10
    val historyLength = 5
    val model = SessionRecommender[Float](itemCount, sessionLength = sessionLength,
      includeHistory = true, historyLength = historyLength)
    val ran = RandomGenerator.RNG
    val data = (1 to 100).map { x =>
      val items1: Seq[Float] = for (i <- 1 to sessionLength) yield
        ran.uniform(1, itemCount).toInt.toFloat
      val items2: Seq[Float] = for (i <- 1 to historyLength) yield
        ran.uniform(1, itemCount).toInt.toFloat
      val input1 = Tensor(items1.toArray, Array(sessionLength)).resize(1, sessionLength)
      val input2 = Tensor(items2.toArray, Array(historyLength)).resize(1, historyLength)
      T(input1, input2)
    }
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }
  }

  "SessionRecommender recommendForSession" should "work properly" in {
    val itemCount = 100
    val sessionLength = 10
    val historyLength = 5
    val model = SessionRecommender[Float](itemCount, sessionLength = sessionLength,
      includeHistory = true, historyLength = historyLength)
    val ran = RandomGenerator.RNG
    val data1: Array[Sample[Float]] = (1 to 10)
      .map { x =>
        val items1: Seq[Float] = for (i <- 1 to sessionLength) yield
          ran.uniform(1, itemCount).toInt.toFloat
        val items2: Seq[Float] = for (i <- 1 to historyLength) yield
          ran.uniform(1, itemCount).toInt.toFloat
        val input1 = Tensor(items1.toArray, Array(sessionLength))
        val input2 = Tensor(items2.toArray, Array(historyLength))
        Sample[Float](Array(input1, input2))
      }.toArray

    val recommedations1 = model.recommendForSession(data1, 4, zeroBasedLabel = false)
    recommedations1.map { x =>
      assert(x.size == 4)
      assert(x(0)._2 >= x(1)._2)
    }

    val data2: RDD[Sample[Float]] = sc.parallelize(data1)
    val recommedations2 = model.recommendForSession(data2, 3, zeroBasedLabel = false)
    recommedations2.take(10).map { x =>
      assert(x.size == 3)
      assert(x(0)._2 >= x(1)._2)
    }
  }

  "SessionRecommender compile and fit" should "work properly" in {
    val itemCount = 100
    val sessionLength = 10
    val historyLength = 5
    val model = SessionRecommender[Float](itemCount, 10, sessionLength = sessionLength,
      includeHistory = true, historyLength = historyLength)
    val data1 = sc.parallelize(1 to 100)
      .map { x =>
        val ran = RandomGenerator.RNG
        val items1: Seq[Float] = for (i <- 1 to sessionLength) yield
          ran.uniform(1, itemCount).toInt.toFloat
        val items2: Seq[Float] = for (i <- 1 to historyLength) yield
          ran.uniform(1, itemCount).toInt.toFloat
        val input1 = Tensor(items1.toArray, Array(sessionLength))
        val input2 = Tensor(items2.toArray, Array(historyLength))
        val label = Tensor[Float](1).apply1(_ => ran.uniform(1, itemCount).toInt.toFloat)
        Sample(Array(input1, input2), Array(label))
      }
    model.compile(optimizer = "rmsprop", loss = "sparse_categorical_crossentropy")
    model.summary()
    model.fit(data1, nbEpoch = 1)
  }
}

class SessionRecommenderSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val ran = RandomGenerator.RNG
    val itemCount = 100
    val sessionLength = 10
    val model = SessionRecommender[Float](100, sessionLength = 10)
    val items: Seq[Float] = for (i <- 1 to sessionLength) yield
      ran.uniform(1, itemCount).toInt.toFloat
    val data = Tensor(items.toArray, Array(sessionLength)).resize(1, sessionLength)
    ZooSpecHelper.testZooModelLoadSave(model, data, SessionRecommender.loadModel[Float])
  }
}

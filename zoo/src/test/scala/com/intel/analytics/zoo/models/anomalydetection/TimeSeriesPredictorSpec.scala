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

import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.Vectors
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}


class TimeSeriesPredictorSpec  extends ZooSpecHelper {

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

  "scalar should " should "work properly" in {

    val data = Array(
      Vectors.dense(0.0, 10.3, 1.0, 4.0, 5.0),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    )

    val df = sqlContext.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    val scalerModel = scaler.fit(df)

    scalerModel.transform(df).show(100)

    println(scalerModel.std +"," + scalerModel.mean)


  }

  "TimeSeriesPredictor unroll multiple features " should "work properly" in {
    val data: RDD[Array[Float]] = sc.parallelize(5 to 20)
      .map(x => Array(x.toFloat))

    val unrolled = TimeSeriesPredictor.unroll[Float](data, encoderLength = 4, decoderLength=3)

      unrolled.take(10).foreach(println)

    println(unrolled.count())

    val unrolled2 = AnomalyDetector.unroll[Float](data, unrollLength = 4, predictStep = 1)
    println(unrolled2.count())

  }


}

class TimeSeriesPredictorSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {

    val model = TimeSeriesPredictor[Float](Shape(10, 2), Array(10, 5), Array(0.2f, 0.2f))
    val data = Tensor[Float](10, 10, 2).randn(0.0, 0.1)
    ZooSpecHelper.testZooModelLoadSave(model, data, AnomalyDetector.loadModel[Float])
  }
}

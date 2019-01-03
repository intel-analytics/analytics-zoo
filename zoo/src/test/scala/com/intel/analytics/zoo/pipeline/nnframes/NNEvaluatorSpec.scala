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
package com.intel.analytics.zoo.pipeline.nnframes

import com.intel.analytics.bigdl.models.inception.Inception_v1
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.pipeline.api.keras.metrics.{AUC, AucScore, Top5Accuracy}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class NNEvaluatorSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc : SparkContext = _
  var sqlContext : SQLContext = _
  var smallData: Seq[(Array[Double], Double)] = _
  val nRecords = 100
  val maxEpoch = 20

  before {
    Random.setSeed(42)
    RNG.setSeed(42)
    val conf = Engine.createSparkConf().setAppName("Test NNEstimator").setMaster("local[1]")
    sc = NNContext.initNNContext(conf)
    sqlContext = new SQLContext(sc)
    smallData = NNEstimatorSpec.generateTestInput(
      nRecords, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), -1.0, 42L)
  }

  after{
    if (sc != null) {
      sc.stop()
    }
  }

  "NNEvaluator" should "support single dimension output" in {
    val model = new Sequential().add(Linear[Float](6, 10)).add(Linear[Float](10, 1))
      .add(Sigmoid[Float])
    val criterion = BCECriterion[Float]()
    val classifier = NNClassifier(model, criterion, Array(6))
      .setOptimMethod(new Adam[Float]())
      .setLearningRate(0.01)
      .setBatchSize(10)
      .setMaxEpoch(10)
    val data = sc.parallelize(smallData.map(t => (t._1, t._2 - 1.0)))
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val nnModel = classifier.fit(df)
    val predictionDF = nnModel.transform(df)
    val evaluateResult = new NNEvaluator().evaluate(predictionDF, Array(new AUC()))
    require(evaluateResult.head._1.result()._1 > 0.8)
  }

  "NNEvaluator" should "support image FEATURE types" in {
    val pascalResource = getClass.getClassLoader.getResource("imagenet/n02110063/")
    val imageDF = NNImageReader.readImages(pascalResource.getFile, sc).withColumn("label", lit(2))
    assert(imageDF.count() == 3)
    val transformer = RowToImageFeature() -> ImageResize(256, 256) -> ImageCenterCrop(224, 224) ->
      ImageChannelNormalize(123, 117, 104) -> ImageMatToTensor() -> ImageFeatureToTensor()
    val predictionDF = NNClassifierModel(Inception_v1(1000), transformer)
      .setFeaturesCol("image")
      .transform(imageDF)

    val evaluateResult = new NNEvaluator().evaluate(predictionDF, Array(new Top1Accuracy[Float]()))
    require(evaluateResult.head._1.isInstanceOf[AccuracyResult])
  }

  "NNEvaluator" should "support image FEATURE types with NNEstimator" in {
    val pascalResource = getClass.getClassLoader.getResource("imagenet/n02110063/")
    val imageDF = NNImageReader.readImages(pascalResource.getFile, sc).withColumn("label", lit(2))
    assert(imageDF.count() == 3)
    val transformer = RowToImageFeature() -> ImageResize(256, 256) -> ImageCenterCrop(224, 224) ->
      ImageChannelNormalize(123, 117, 104) -> ImageMatToTensor() -> ImageFeatureToTensor()
    val predictionDF = NNModel(Inception_v1(1000), transformer)
      .setFeaturesCol("image")
      .transform(imageDF)

    val evaluateResult = new NNEvaluator().evaluate(predictionDF, Array(new Top1Accuracy[Float](),
      new Top5Accuracy[Float]()))
    require(evaluateResult.forall(r => r._1.isInstanceOf[AccuracyResult]))
  }
}

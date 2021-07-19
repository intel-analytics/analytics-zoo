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
package com.intel.analytics.zoo.examples.tfpark.sparkEstimator

import com.intel.analytics.bigdl.optim.{Top1Accuracy, Trigger}
import com.intel.analytics.bigdl.utils.LoggerFilter
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.feature.common.{FeatureLabelPreprocessing, ScalarToTensor}
import com.intel.analytics.zoo.feature.image.{RowToImageFeature, _}
import com.intel.analytics.zoo.pipeline.nnframes.NNImageReader
import com.intel.analytics.zoo.tfpark.SparkEstimator
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.SparkConf
import scopt.OptionParser

/**
 * Scala example for image fine tuning with Inception model on Spark DataFrame.
 * Please refer to the readme.md in the same folder for more details.
 */
object Main {

  LoggerFilter.redirectSparkInfoLogs()

  def main(args: Array[String]): Unit = {
    Utils.trainParser.parse(args, Utils.TrainParams()).foreach(param => {
      val conf = new SparkConf().setAppName("Spark Estimator Example")
      val sc = NNContext.initNNContext(conf)

      val createLabel = udf { row: Row =>
        if (new Path(row.getString(0)).getName.contains("cat")) 1.0 else 2.0
      }
      val imagesDF = NNImageReader.readImages(param.imagePath, sc,
          resizeH = 256, resizeW = 256, imageCodec = 1)
        .withColumn("label", createLabel(col("image")))
      val Array(validationDF, trainingDF) = imagesDF.randomSplit(Array(0.20, 0.80), seed = 1L)

      val featureTransformer = RowToImageFeature() ->
        ImageCenterCrop(224, 224) -> ImageChannelNormalize(123, 117, 104) ->
        ImageMatToTensor() -> ImageFeatureToTensor()

      val classifier = SparkEstimator(param.modelPath)
        .setSamplePreprocessing(FeatureLabelPreprocessing(featureTransformer, ScalarToTensor()))
        .setFeaturesCol("image")
        .setBatchSize(param.batchSize)
        .setMaxEpoch(param.nEpochs)
        .setCachingSample(false)
        .setValidation(Trigger.everyEpoch, validationDF, Array(new Top1Accuracy()), param.batchSize)

      val pipeline = new Pipeline().setStages(Array(classifier))
      val pipelineModel = pipeline.fit(trainingDF)
      val predictions = pipelineModel.transform(validationDF)

      predictions.select("image", "label", "prediction").sample(false, 0.05).show(false)
      sc.stop()
    })
  }
}


private object Utils {

  case class TrainParams(
      modelPath: String = "/tmp/zoo/tfpark/inceptionV1_models",
      imagePath: String = "/tmp/zoo/dogs_cats/samples",
      batchSize: Int = 32,
      nEpochs: Int = 2)

  val trainParser = new OptionParser[TrainParams]("TFPark SparkEstimator Example") {
    opt[String]('m', "modelPath")
      .text("pretrained model path")
      .action((x, c) => c.copy(modelPath = x))
    opt[String]('d', "imagePath")
      .text("training data path")
      .action((x, c) => c.copy(imagePath = x))
    opt[Int]('b', "batchSize")
      .text("batchSize")
      .action((x, c) => c.copy(batchSize = x))
    opt[Int]('e', "nEpochs")
      .text("epoch numbers")
      .action((x, c) => c.copy(nEpochs = x))
  }
}

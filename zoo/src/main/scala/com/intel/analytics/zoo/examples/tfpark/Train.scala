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

package com.intel.analytics.zoo.examples.tfpark

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToSample}
import com.intel.analytics.bigdl.models.lenet.Utils.{trainMean, trainStd}
import com.intel.analytics.bigdl.optim.{Adam, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.LoggerFilter
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.tfpark.TFOptimizer
import scopt.OptionParser

object Train {

  LoggerFilter.redirectSparkInfoLogs()

  case class TrainParam(dataPath: String = "/tmp/mnist",
                        modelPath: String = "/tmp/lenet_export",
                        batchSize: Int = 280,
                        numEpoch: Int = 5)

  val parser = new OptionParser[TrainParam]("TFNet Train Example") {
    head("TFNet Train Example")
    opt[String]('d', "data")
      .text("The path where the data are stored, can be a folder")
      .action((x, c) => c.copy(dataPath = x))
    opt[String]("model")
      .text("The path of the TensorFlow model folder")
      .action((x, c) => c.copy(modelPath = x))
    opt[Int]('b', "batchSize")
      .text("The batchSize")
      .action((x, c) => c.copy(batchSize = x))
    opt[Int]('e', "epoch")
      .text("number of epochs")
      .action((x, c) => c.copy(numEpoch = x))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, TrainParam()).foreach { params =>
      val sc = NNContext.initNNContext("TFPark Training Example")

      val trainData = params.dataPath + "/train-images-idx3-ubyte"
      val trainLabel = params.dataPath + "/train-labels-idx1-ubyte"
      val validationData = params.dataPath + "/t10k-images-idx3-ubyte"
      val validationLabel = params.dataPath + "/t10k-labels-idx1-ubyte"

      val trainRDD = sc.parallelize(load(trainData, trainLabel))
      val trainDataSet = FeatureSet.rdd(trainRDD) ->
        BytesToGreyImg(28, 28) ->
        GreyImgNormalizer(trainMean * 255, trainStd * 255) ->
        GreyImgToSample() ->
        new SampleToTFTrainSample() ->
        SampleToMiniBatch(params.batchSize)

      val optimizer = new TFOptimizer(params.modelPath,
        new Adam[Float](1e-3),
        trainDataSet)
      optimizer.optimize(Trigger.maxEpoch(params.numEpoch))
    }
  }


  private def load(featureFile: String, labelFile: String): Array[ByteRecord] = {

    val featureBuffer =
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    val labelBuffer =
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)))

    val labelMagicNumber = labelBuffer.getInt()

    require(labelMagicNumber == 2049)
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)

    val rowNum = featureBuffer.getInt()
    val colNum = featureBuffer.getInt()

    val result = new Array[ByteRecord](featureCount)
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum))
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = ByteRecord(img, labelBuffer.get().toFloat + 1.0f)
      i += 1
    }

    result
  }

  class SampleToTFTrainSample() extends Transformer[Sample[Float], Sample[Float]] {

    override def apply(prev: Iterator[Sample[Float]]): Iterator[Sample[Float]] = {
      prev.map(sample => {
        // tf model needs feature shape (28,28,1)
        val input1 = sample.feature().reshape(Array(28, 28, 1))
        // tf model needs label as scalar and starting from 0
        val input2 = Tensor.scalar[Float](sample.label().valueAt(1) - 1)

        ArraySample(Array[Tensor[Float]](input1, input2), Tensor[Float]().resize(1).zero())
      })
    }
  }
}


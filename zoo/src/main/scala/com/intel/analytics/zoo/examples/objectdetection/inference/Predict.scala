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

package com.intel.analytics.zoo.examples.objectdetection.inference

import java.nio.file.Paths

import com.intel.analytics.zoo.common.{NNContext, Utils}
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.models.image.objectdetection.{ObjectDetector, Visualizer}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.opencv.imgcodecs.Imgcodecs
import scopt.OptionParser

object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class PredictParam(image: String = "",
    outputFolder: String = "data/demo",
    modelPath: String = "",
    nPartition: Int = 1)

  val parser = new OptionParser[PredictParam]("Analytics Zoo Object Detection Demo") {
    head("Analytics Zoo Object Detection Demo")
    opt[String]('i', "image")
      .text("where you put the demo image data, can be image folder or image path")
      .action((x, c) => c.copy(image = x))
      .required()
    opt[String]('o', "output")
      .text("where you put the output data")
      .action((x, c) => c.copy(outputFolder = x))
      .required()
    opt[String]("modelPath")
      .text("Analytics Zoo model path")
      .action((x, c) => c.copy(modelPath = x))
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
      .required()
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, PredictParam()).foreach { params =>
      val conf = new SparkConf()
        .setAppName("Object Detection Inference Example")
      val sc = NNContext.initNNContext(conf)

      val model = ObjectDetector.loadModel[Float](params.modelPath)
      val data = ImageSet.read(params.image, sc, params.nPartition,
        imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR)
      val output = model.predictImageSet(data)

      val visualizer = Visualizer(model.getConfig.labelMap, encoding = "jpg")
      val visualized = visualizer(output).toDistributed()
      val result = visualized.rdd.map(imageFeature =>
        (imageFeature.uri(), imageFeature[Array[Byte]](Visualizer.visualized))).collect()

      result.foreach(x => {
        Utils.saveBytes(x._2, getOutPath(params.outputFolder, x._1, "jpg"), true)
      })
      logger.info(s"labeled images are saved to ${params.outputFolder}")
    }
  }

  private def getOutPath(outPath: String, uri: String, encoding: String): String = {
    Paths.get(outPath,
      s"detection_${ uri.substring(uri.lastIndexOf("/") + 1,
        uri.lastIndexOf(".")) }.${encoding}").toString
  }
}

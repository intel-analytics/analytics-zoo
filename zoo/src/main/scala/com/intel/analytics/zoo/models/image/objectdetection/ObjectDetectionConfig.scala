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

package com.intel.analytics.zoo.models.image.objectdetection

import com.intel.analytics.bigdl.dataset.PaddingParam
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.feature.common.Preprocessing
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.models.image.common.ImageConfigure
import com.intel.analytics.zoo.models.image.objectdetection.ObjectDetectorDataset.{Coco, Pascal}

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.tensor.Tensor

private[models] object ObjectDetectionConfig {

  val models = Set("ssd-vgg16-300x300",
    "ssd-vgg16-300x300-quantize",
    "ssd-vgg16-512x512",
    "ssd-vgg16-512x512-quantize",
    "ssd-mobilenet-300x300",
    "frcnn-vgg16",
    "frcnn-vgg16-compress",
    "frcnn-vgg16-quantize",
    "frcnn-vgg16-compress-quantize",
    "frcnn-pvanet",
    "frcnn-pvanet-quantize",
    "frcnn-pvanet-compress",
    "frcnn-pvanet-compress-quantize",
    "frcnn-vgg16-compress-quantize")

  def apply[T: ClassTag](model: String, dataset: String, version: String)
    (implicit ev: TensorNumeric[T]): ImageConfigure[T] = {
    val labelMap = LabelReader(dataset)
    model match {
      case "ssd-vgg16-300x300" |
           "ssd-vgg16-300x300-quantize" =>
        ImageConfigure(ObjectDetectionConfig.preprocessSsdVgg(300, dataset, version),
          ScaleDetection(),
          batchPerPartition = 2,
          labelMap = labelMap)
      case "ssd-vgg16-512x512" |
           "ssd-vgg16-512x512-quantize" =>
        ImageConfigure(ObjectDetectionConfig.preprocessSsdVgg(512, dataset, version),
          ScaleDetection(),
          batchPerPartition = 2,
          labelMap = labelMap)
      case "ssd-mobilenet-300x300" =>
        ImageConfigure(ObjectDetectionConfig.preprocessSsdMobilenet(300, dataset, version),
          ScaleDetection(),
          batchPerPartition = 2,
          labelMap = labelMap
        )
      case "frcnn-vgg16" |
           "frcnn-vgg16-quantize" |
           "frcnn-vgg16-compress" |
           "frcnn-vgg16-compress-quantize" =>
        ImageConfigure(ObjectDetectionConfig.preprocessFrcnnVgg(dataset, version),
          DecodeOutput(),
          batchPerPartition = 1,
          labelMap,
          Some(PaddingParam()))
      case "frcnn-pvanet" |
           "frcnn-pvanet-quantize" |
           "frcnn-pvanet-compress" |
           "frcnn-pvanet-compress-quantize" =>
        ImageConfigure(ObjectDetectionConfig.preprocessFrcnnPvanet(dataset, version),
          DecodeOutput(),
          batchPerPartition = 1,
          labelMap,
          Some(PaddingParam()))
    }
  }

  def preprocessSsdVgg(resolution: Int, dataset: String, version: String)
  : Preprocessing[ImageFeature, ImageFeature] = {
    preprocessSsd(resolution, (123f, 117f, 104f), 1f)
  }

  def preprocessSsd(resolution: Int, meansRGB: (Float, Float, Float),
    scale: Float): Preprocessing[ImageFeature, ImageFeature] = {
    ImageResize(resolution, resolution) ->
      ImageChannelNormalize(meansRGB._1, meansRGB._2, meansRGB._3, scale, scale, scale) ->
      ImageMatToTensor() -> ImageSetToSample()
  }

  def preprocessSsdMobilenet(resolution: Int, dataset: String, version: String)
  : Preprocessing[ImageFeature, ImageFeature] = {
    ObjectDetectorDataset(dataset) match {
      case Pascal =>
        preprocessSsd(resolution, (127.5f, 127.5f, 127.5f), 1 / 0.007843f)
      case Coco =>
        throw new Exception("coco is not yet supported for Analytics Zoo ssd mobilenet")
    }
  }

  def preprocessFrcnn(resolution: Int, scaleMultipleOf: Int):
    Preprocessing[ImageFeature, ImageFeature] = {
    ImageAspectScale(resolution, scaleMultipleOf) ->
      ImageChannelNormalize(122.7717f, 115.9465f, 102.9801f) ->
      ImageMatToTensor() -> ImInfo() -> DummyGT() ->
      ImageSetToSample(Array(ImageFeature.imageTensor, "ImInfo", "GT"))
  }

  def preprocessFrcnnVgg(dataset: String, version: String):
    Preprocessing[ImageFeature, ImageFeature] = {
    ObjectDetectorDataset(dataset) match {
      case Pascal =>
        preprocessFrcnn(600, 1)
      case Coco =>
        throw new Exception("coco is not yet supported for Analytics Zoo FrcnnVgg")
    }
  }

  def preprocessFrcnnPvanet(dataset: String, version: String):
    Preprocessing[ImageFeature, ImageFeature] = {
    ObjectDetectorDataset(dataset) match {
      case Pascal =>
        preprocessFrcnn(640, 32)
      case Coco =>
        throw new Exception("coco is not yet supported for BigDL FrcnnPvanet")
    }
  }
}

sealed trait ObjectDetectorDataset {
  val value: String
}

object ObjectDetectorDataset {
  def apply(datasetString: String): ObjectDetectorDataset = {
    datasetString.toUpperCase match {
      case Pascal.value => Pascal
      case Coco.value => Coco
    }
  }

  case object Pascal extends ObjectDetectorDataset {
    val value = "PASCAL"
  }

  case object Coco extends ObjectDetectorDataset {
    val value: String = "COCO"
  }
}

/**
 * Generate imInfo
 * imInfo is a tensor that contains height, width, scaleInHeight, scaleInWidth
 */
case class ImInfo() extends ImageProcessing {
  override def transformMat(feature: ImageFeature): Unit = {
    feature("ImInfo") = feature.getImInfo()
  }
}

case class DummyGT() extends ImageProcessing {
  override def transformMat(feature: ImageFeature): Unit = {
    feature("DummyGT") = Tensor[Float](1)
  }
}

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

package com.intel.analytics.zoo.models.image.common

import com.intel.analytics.bigdl.dataset.PaddingParam
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature}
import com.intel.analytics.zoo.feature.common.{Preprocessing}
import com.intel.analytics.zoo.models.image.objectdetection.ObjectDetectionConfig
import com.intel.analytics.zoo.models.image.imageclassification.ImageClassificationConfig
import org.apache.log4j.Logger

import scala.reflect.ClassTag

/**
 * predictor configure
 *
 * @param preProcessor preprocessor of ImageSet before model inference
 * @param postProcessor postprocessor of ImageSet after model inference
 * @param batchPerPartition batch size per partition
 * @param labelMap label mapping
 * @param featurePaddingParam featurePaddingParam if the inputs have variant size
 */
case class ImageConfigure[T: ClassTag](
  preProcessor: Preprocessing[ImageFeature, ImageFeature] = null,
  postProcessor: Preprocessing[ImageFeature, ImageFeature] = null,
  batchPerPartition: Int = 4,
  labelMap: Map[Int, String] = null,
  featurePaddingParam: Option[PaddingParam[T]] = None)(implicit ev: TensorNumeric[T]) {
}

object ImageConfigure {
  val logger = Logger.getLogger(getClass)

  val splitter = "_"

  /**
   * Get config for each model
   *
   * @param tag If model is published by analytics zoo, tag should be in
   *            'publisher_model_dataset_version' format,
   *            and publisher is required to be 'analytics-zoo'.
   *            If model is not published by analytics zoo, tag does not
   *            need to start with 'analytics-zoo', and format does not need to
   *            be in 'publisher_model_dataset_version' format
   * @return
   */
  def parse[T: ClassTag](tag: String)(implicit ev: TensorNumeric[T]): ImageConfigure[T] = {
    // If tag does not start with "analytics-zoo", it's a third-party model.
    // Don't create default image configuration
    if (!tag.startsWith("analytics-zoo")) {
      logger.warn("Only support to create default image configuration for models published by " +
        "analytics zoo. Third-party models need to pass its own image configuration during " +
        "prediction")
      null
    } else {
      val splits = tag.split(splitter)
      // It's a analytics zoo model. The tag should be in 'publisher_model_dataset_version' format
      require(splits.length >= 4, s"tag ${tag}" +
        s" needs at least 4 elements, publisher, model, dataset, version")
      val model = splits(1)
      val dataset = splits(2)
      val version = splits(3)
      model.toLowerCase() match {
        case obModel if ObjectDetectionConfig.models contains obModel =>
          ObjectDetectionConfig(obModel, dataset, version)
        case imcModel if ImageClassificationConfig.models contains imcModel =>
          ImageClassificationConfig(imcModel, dataset, version)
        case _ => logger.warn(s"$model is not defined in Analytics zoo.")
          null
      }
    }
  }
}

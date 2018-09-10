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

package com.intel.analytics.zoo.models.image.objectdetection.common

import java.io.File

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.label.roi._
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, ImageFeature, MatToFloats}
import com.intel.analytics.zoo.feature.image.{ImageSet, LocalImageSet}
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.{FrcnnMiniBatch,
  FrcnnToBatch}
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.roiimage.{ByteRecord,
  RecordToFeature, RoiImageToBatch, SSDMiniBatch}
import com.intel.analytics.zoo.models.image.objectdetection.fasterrcnn
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag


object IOUtils {
  def loadSeqFiles(nPartition: Int, seqFloder: String, sc: SparkContext): RDD[ByteRecord] = {
    sc.sequenceFile(seqFloder, classOf[Text], classOf[Text],
      nPartition).map(x => ByteRecord(x._2.copyBytes(), x._1.toString))
  }

  def localImagePaths(folder: String): LocalImageSet = {
    val arr = new File(folder).listFiles().map(x => {
      val imf = ImageFeature()
      imf(ImageFeature.uri) = x.getAbsolutePath
      imf
    })
    ImageSet.array(arr)
  }

  def loadSSDTrainSet(folder: String, sc: SparkContext, resolution: Int, batchSize: Int,
                      parNum: Int)
  : DataSet[SSDMiniBatch] = {
    val trainRdd = loadSeqFiles(parNum, folder, sc)
    DataSet.rdd(trainRdd) -> RecordToFeature(true) ->
      BytesToMat() ->
      RoiNormalize() ->
      ColorJitter() ->
      RandomTransformer(Expand() -> RoiProject(), 0.5) ->
      RandomSampler() ->
      Resize(resolution, resolution, -1) ->
      RandomTransformer(HFlip() -> RoiHFlip(), 0.5) ->
      ChannelNormalize(123f, 117f, 104f) ->
      MatToFloats(validHeight = resolution, validWidth = resolution) ->
      RoiImageToBatch(batchSize)
  }

  def loadSSDValSet(folder: String, sc: SparkContext, resolution: Int, batchSize: Int, parNum: Int)
  : DataSet[SSDMiniBatch] = {
    val valRdd = loadSeqFiles(parNum, folder, sc)

    DataSet.rdd(valRdd) -> RecordToFeature(true) ->
      BytesToMat() ->
      RoiNormalize() ->
      Resize(resolution, resolution) ->
      ChannelNormalize(123f, 117f, 104f) ->
      MatToFloats(validHeight = resolution, validWidth = resolution) ->
      RoiImageToBatch(batchSize)
  }

  def loadFasterrcnnTrainSet(folder: String, sc: SparkContext, param: fasterrcnn.PreProcessParam,
                             batchSize: Int, parNum: Int)
  : DataSet[FrcnnMiniBatch] = {
    val trainRdd = loadSeqFiles(parNum, folder, sc)
    DataSet.rdd(trainRdd) -> RecordToFeature(true) ->
      BytesToMat() ->
      RandomAspectScale(param.scales, param.scaleMultipleOf) -> RoiResize() ->
      RandomTransformer(HFlip() -> RoiHFlip(false), 0.5) ->
      ChannelNormalize(param.pixelMeanRGB._1, param.pixelMeanRGB._2, param.pixelMeanRGB._3) ->
      MatToFloats(validHeight = 600, validWidth = 600) ->
      FrcnnToBatch(batchSize, true)
  }

  def loadFasterrcnnValSet(folder: String, sc: SparkContext, param: fasterrcnn.PreProcessParam,
                           batchSize: Int, parNum: Int)
  : DataSet[FrcnnMiniBatch] = {
    val valRdd = loadSeqFiles(parNum, folder, sc)

    DataSet.rdd(valRdd) -> RecordToFeature(true) ->
      BytesToMat() ->
      AspectScale(param.scales(0), param.scaleMultipleOf) ->
      ChannelNormalize(param.pixelMeanRGB._1, param.pixelMeanRGB._2, param.pixelMeanRGB._3) ->
      MatToFloats(100, 100) ->
      FrcnnToBatch(param.batchSize, true)
  }
}

object OBUtils {
  def addConvRelu[@specialized(Float, Double) T: ClassTag](prevNodes: ModuleNode[T],
    p: (Int, Int, Int, Int, Int), name: String, prefix: String = "conv", nGroup: Int = 1,
    propogateBack: Boolean = true)(implicit ev: TensorNumeric[T]): ModuleNode[T] = {
    val conv = SpatialConvolution[T](p._1, p._2, p._3, p._3, p._4, p._4,
      p._5, p._5, nGroup = nGroup, propagateBack = propogateBack)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
      .setName(s"$prefix$name").inputs(prevNodes)
    ReLU[T](true).setName(s"relu$name").inputs(conv)
  }

  def stopGradient[@specialized(Float, Double) T: ClassTag](model: Graph[T]): Unit = {
    val priorboxNames = model.modules
      .filter(_.getClass.getName.toLowerCase().endsWith("priorbox"))
      .map(_.getName()).toArray
    model.stopGradient(priorboxNames)
  }
}

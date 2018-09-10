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

package com.intel.analytics.zoo.models.image.objectdetection.fasterrcnn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.models.image.objectdetection.common.ModuleUtil

object FasterRcnn {
  /**
   *
   * @param p parameter: (nIn: Int, nOut: Int, ker: Int, stride: Int, pad: Int)
   * @param name name of layer
   * @return
   */
  def conv(p: (Int, Int, Int, Int, Int),
    name: String, withBias: Boolean = true): SpatialConvolution[Float] = {
    SpatialConvolution(p._1, p._2, p._3, p._3, p._4, p._4, p._5, p._5,
      withBias = withBias).setName(name)
  }


  def apply(nClass: Int, rpnPreNmsTopNTest: Int, rpnPostNmsTopNTest: Int,
    ratios: Array[Float], scales: Array[Float],
    rpn: Sequential[Float], fastRcnn: Sequential[Float],
    rpnPreNmsTopNTrain: Int = 12000, rpnPostNmsTopNTrain: Int = 2000): Module[Float] = {
    val model = Sequential()
    val model1 = ParallelTable()
    model1.add(rpn)
    model1.add(new Identity())
    model.add(model1)
    // connect rpn and fast-rcnn
    val middle = ConcatTable()
    val left = Sequential()
    val left1 = ConcatTable()
    left1.add(selectTensor(1, 1, 1))
    left1.add(selectTensor(1, 1, 2))
    left1.add(selectTensor1(2))
    left.add(left1)
    left.add(Proposal(rpnPreNmsTopNTest,
      rpnPostNmsTopNTest,
      ratios, scales, rpnPreNmsTopNTrain, rpnPostNmsTopNTrain))
    // first add feature from feature net
    middle.add(selectTensor(1, 2))
    // then add rois from proposal
    middle.add(left)
    model.add(middle)
    // get the fast rcnn results and rois
    model.add(ConcatTable().add(fastRcnn).add(selectTensor(2)))
    ModuleUtil.shareMemory(model)
    model
  }


  /**
   * select tensor from nested tables
   * @param depths a serious of depth to use when fetching certain tensor
   * @return a wanted tensor
   */
  protected def selectTensor(depths: Int*): Sequential[Float] = {
    val module = new Sequential[Float]()
    var i = 0
    while (i < depths.length - 1) {
      module.add(new SelectTable(depths(i)))
      i += 1
    }
    module.add(new SelectTable(depths(depths.length - 1)))
  }

  protected def selectTensor1(depth: Int): SelectTable[Float] = {
    new SelectTable(depth)
  }
}

/**
 * Preprocess parameters
 * @param batchSize should be 1
 * @param scales Each scale is the pixel size of an image"s shortest side, can contain multiple
 * @param scaleMultipleOf Resize test images so that its width and height are multiples of ...
 * @param pixelMeanRGB mean value to be sub from
 * @param hasLabel whether data contains label, default is false
 */
case class PreProcessParam(batchSize: Int = 1,
  scales: Array[Int] = Array(600), scaleMultipleOf: Int = 1,
  pixelMeanRGB: (Float, Float, Float) = (122.7717f, 115.9465f, 102.9801f),
  hasLabel: Boolean = false, nPartition: Int = -1, norms: (Float, Float, Float) = (1f, 1f, 1f)
)

/**
 * post process parameters
 * @param nmsThresh Overlap threshold used for non-maximum suppression (suppress boxes with
 * IoU >= this threshold)
 * @param nClasses number of classes
 * @param bboxVote whether apply bounding box voting
 * @param maxPerImage
 * @param thresh
 */
case class PostProcessParam(nmsThresh: Float = 0.3f, nClasses: Int,
  bboxVote: Boolean, maxPerImage: Int = 100, thresh: Double = 0.05)

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

package com.intel.analytics.zoo.models.image.objectdetection.ssd

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Utils => BUtils, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{L2Regularizer, Regularizer}
import com.intel.analytics.bigdl.tensor.Storage
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.models.image.objectdetection.common.ModuleUtil
import com.intel.analytics.bigdl.nn.DetectionOutputSSD
import com.intel.analytics.bigdl.utils.Node
import org.apache.log4j.Logger

import scala.reflect.ClassTag

/**
 *
 * @param nInput
 * @param nPriors
 * @param minSizes
 * @param maxSizes
 * @param aspectRatios
 * @param isFlip
 * @param isClip
 * @param variances
 * @param step
 */
case class ComponetParam(nInput: Int, nPriors: Int, minSizes: Array[Float],
  maxSizes: Array[Float], aspectRatios: Array[Float], isFlip: Boolean,
  isClip: Boolean, variances: Array[Float], step: Int = 0, resolution: Int = 300)

case class PreProcessParam(batchSize: Int = 4,
  resolution: Int = 300,
  pixelMeanRGB: (Float, Float, Float),
  hasLabel: Boolean, nPartition: Int,
  norms: (Float, Float, Float) = (1f, 1f, 1f)
)

object SSDGraph {

  def apply[@specialized(Float, Double) T: ClassTag](numClasses: Int, resolution: Int,
    input: ModuleNode[T], basePart1: ModuleNode[T], basePart2: ModuleNode[T],
    params: Map[String, ComponetParam], isLastPool: Boolean, normScale: Float,
                                                     shareLocation: Boolean = true,
                                                     bgLabel: Int = 0,
                                                     nmsThresh: Float = 0.45f,
                                                     nmsTopk: Int = 400,
                                                     keepTopK: Int = 200,
                                                     confThresh: Float = 0.01f,
                                                     varianceEncodedInTarget: Boolean = false,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null)(implicit ev: TensorNumeric[T])
  : AbstractModule[Activity, Activity, T] = {

    val _wRegularizer = if (wRegularizer == null) L2Regularizer[T](0.0005) else wRegularizer
    val conv4_3_norm = NormalizeScale(2, scale = normScale,
      size = Array(1, params("conv4_3_norm").nInput, 1, 1),
      wRegularizer = L2Regularizer[T](0.0005))
      .setName("conv4_3_norm").inputs(basePart1)

    val base2 = basePart2
    val fc6 = SpatialDilatedConvolution[T](params("fc7").nInput,
      1024, 3, 3, 1, 1, 6, 6, 6, 6)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
      .setName("fc6").inputs(base2)

    val relu6 = ReLU[T](true).setName("relu6").inputs(fc6)
    val fc7 = SpatialConvolution[T](1024, 1024, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros).setName("fc7").inputs(relu6)
    val relu7 = ReLU[T](true).setName("relu7").inputs(fc7)

    val norm4_3Out = getConcatOutput(conv4_3_norm, "conv4_3_norm", params,
      params("conv4_3_norm").nInput, numClasses)
    val fc7Out = getConcatOutput(relu7, "fc7", params, 1024, numClasses)

    val relu6_1 = ModuleUtil.addConvRelu(relu7, (1024, 256, 1, 1, 0), "6_1")
    val relu6_2 = ModuleUtil.addConvRelu(relu6_1, (256, 512, 3, 2, 1), "6_2")
    val c6Out = getConcatOutput(relu6_2, "conv6_2", params, params("conv6_2").nInput, numClasses)

    val relu7_1 = ModuleUtil.addConvRelu(relu6_2, (512, 128, 1, 1, 0), "7_1")
    val relu7_2 = ModuleUtil.addConvRelu(relu7_1, (128, 256, 3, 2, 1), "7_2")
    val c7Out = getConcatOutput(relu7_2, "conv7_2", params, params("conv7_2").nInput, numClasses)

    val relu8_1 = ModuleUtil.addConvRelu(relu7_2, (256, 128, 1, 1, 0), "8_1")
    val relu8_2 = if (isLastPool || resolution == 512) {
      ModuleUtil.addConvRelu(relu8_1, (128, 256, 3, 2, 1), "8_2")
    } else {
      ModuleUtil.addConvRelu(relu8_1, (128, 256, 3, 1, 0), "8_2")
    }

    val c8Out = getConcatOutput(relu8_2, "conv8_2", params, params("conv8_2").nInput, numClasses)
    val (c9Out, relu9_2) = if (isLastPool) {
      //      addFeatureComponentPool6(com8, params, numClasses)
      val pool6 = SpatialAveragePooling[T](3, 3).setName("pool6").inputs(relu8_2)
      (getConcatOutput(pool6, "pool6", params, params("pool6").nInput, numClasses), pool6)
    } else {
      val relu9_1 = ModuleUtil.addConvRelu(relu8_2, (256, 128, 1, 1, 0), "9_1")
      val relu9_2 = if (resolution == 512) {
        ModuleUtil.addConvRelu(relu9_1, (128, 256, 3, 2, 1), "9_2")
      } else {
        ModuleUtil.addConvRelu(relu9_1, (128, 256, 3, 1, 0), "9_2")
      }
      (getConcatOutput(relu9_2, "conv9_2", params, params("conv9_2").nInput, numClasses), relu9_2)
    }

    val c10Out = if (resolution == 512) {
      val relu10_1 = ModuleUtil.addConvRelu(relu9_2, (256, 128, 1, 1, 0), "10_1")
      val relu10_2 = ModuleUtil.addConvRelu(relu10_1, (128, 256, 4, 1, 1), "10_2")
      getConcatOutput(relu10_2, "conv10_2", params, params("conv10_2").nInput, numClasses)
    } else {
      null
    }

   val (conf, loc, priors) = if (resolution == 300) {
      val conf = JoinTable[T](2, 0)
        .inputs(norm4_3Out._1, fc7Out._1, c6Out._1, c7Out._1, c8Out._1, c9Out._1)
      val loc = JoinTable[T](2, 0)
        .inputs(norm4_3Out._2, fc7Out._2, c6Out._2, c7Out._2, c8Out._2, c9Out._2)
      val priors = JoinTable[T](3, 0)
        .inputs(norm4_3Out._3, fc7Out._3, c6Out._3, c7Out._3, c8Out._3, c9Out._3)
      (conf, loc, priors)
    } else {
      val conf = JoinTable[T](2, 0)
        .inputs(norm4_3Out._1, fc7Out._1, c6Out._1, c7Out._1, c8Out._1, c9Out._1, c10Out._1)
      val loc = JoinTable[T](2, 0)
        .inputs(norm4_3Out._2, fc7Out._2, c6Out._2, c7Out._2, c8Out._2, c9Out._2, c10Out._2)
      val priors = JoinTable[T](3, 0)
        .inputs(norm4_3Out._3, fc7Out._3, c6Out._3, c7Out._3, c8Out._3, c9Out._3, c10Out._3)
      (conf, loc, priors)
    }

    val out = new DetectionOutputSSD[T](numClasses, shareLocation,
      bgLabel,
      nmsThresh,
      nmsTopk,
      keepTopK,
      confThresh,
      varianceEncodedInTarget).setName("detection_out").inputs(Array(loc, conf, priors))
    val model = Graph(input, out)
    ModuleUtil.setScaleB(model, 2, "detection_out")
    ModuleUtil.stopGradient(model)
    setRegularizer(model, _wRegularizer, bRegularizer)
    model
  }


  private def setRegularizer[@specialized(Float, Double) T: ClassTag]
    (module: AbstractModule[Activity, Activity, T],
    wRegularizer: Regularizer[T], bRegularizer: Regularizer[T])
  : Unit = {
    if (module.isInstanceOf[Container[Activity, Activity, T]]) {
      val ms = module.asInstanceOf[Container[Activity, Activity, T]].modules
      ms.foreach(m => setRegularizer(m, wRegularizer, bRegularizer))
    } else if (module.isInstanceOf[SpatialConvolution[T]]) {
      val m = module.asInstanceOf[SpatialConvolution[T]]
      m.wRegularizer = wRegularizer
      m.bRegularizer = bRegularizer
    } else if (module.isInstanceOf[SpatialDilatedConvolution[T]]) {
      val m = module.asInstanceOf[SpatialDilatedConvolution[T]]
      m.wRegularizer = wRegularizer
      m.bRegularizer = bRegularizer
    } else if (module.isInstanceOf[NormalizeScale[T]]) {
      val m = module.asInstanceOf[NormalizeScale[T]]
      m.wRegularizer = wRegularizer
    }
  }

  private def getConcatOutput[@specialized(Float, Double) T: ClassTag](conv: ModuleNode[T],
    name: String, params: Map[String, ComponetParam], nInput: Int, numClasses: Int)
    (implicit ev: TensorNumeric[T]): (ModuleNode[T], ModuleNode[T], ModuleNode[T]) = {
    val conFlat = getLocConfComponent(conv, name, nInput, params(name).nPriors * numClasses,
      "conf")
    val locFlat = getLocConfComponent(conv, name, nInput, params(name).nPriors * 4, "loc")
    val priorBox = getPriorBox(conv, name, params)
    (conFlat, locFlat, priorBox)
  }

  private def getLocConfComponent[@specialized(Float, Double) T: ClassTag](prev: ModuleNode[T],
    name: String, nInput: Int, nOutput: Int, typeName: String)(implicit ev: TensorNumeric[T]):
    ModuleNode[T] = {
    val conv = SpatialConvolution[T](nInput, nOutput, 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
      .setName(s"${ name }_mbox_$typeName").inputs(prev)
    val trans = Transpose[T](Array((2, 3), (3, 4))).setName(s"${ name }_mbox_${ typeName }_perm")
      .inputs(conv)
    InferReshape[T](Array(0, -1)).setName(s"${ name }_mbox_${ typeName }_flat").inputs(trans)
  }

  private def getPriorBox[@specialized(Float, Double) T: ClassTag](conv: ModuleNode[T],
    name: String, params: Map[String, ComponetParam])(implicit ev: TensorNumeric[T]):
    ModuleNode[T] = {
    val param = params(name)
    PriorBox[T](minSizes = param.minSizes, maxSizes = param.maxSizes,
      _aspectRatios = param.aspectRatios, isFlip = param.isFlip, isClip = param.isClip,
      variances = param.variances, step = param.step, offset = 0.5f,
      imgH = param.resolution, imgW = param.resolution)
      .setName(s"${ name }_mbox_priorbox").inputs(conv)
  }

  private val logger = Logger.getLogger(getClass)
}

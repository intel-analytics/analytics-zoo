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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import org.apache.log4j.Logger

import scala.reflect.ClassTag

object ModuleUtil {

  private val logger = Logger.getLogger(getClass)
  val shareFinput = Tensor[Float](1)

  /**
   * share the storage of SpatialConvolution fInput
   * note that this sharing only works for Inference only
   * @param model model to share
   */
  def shareMemory(model: Module[Float], isShareOutput: Boolean = false): Unit = {
    logger.info(s"Share memory in ${ model.getName() }")
    shareFInput(model, shareFinput)
    if (isShareOutput) {
      shareOutput(model)
    }
  }

  private def shareFInput(module: Module[Float], shareFinput: Tensor[Float]): Unit = {
    Utils.getNamedModules(module)
    module match {
      case m: Container[_, _, Float] =>
        for (m <- module.asInstanceOf[Container[_, _, Float]].modules) {
          shareFInput(m, shareFinput)
        }
      case _ =>
        if (module.getClass.getName.endsWith("SpatialConvolution")) {
          module.asInstanceOf[SpatialConvolution[Float]].fInput.set(shareFinput)
        }
    }
  }

  private val out1 = Tensor[Float](1)
  private val out2 = Tensor[Float](1)

  private def shareOutput(module: Module[Float]): Unit = {
    if (module.isInstanceOf[Graph[Float]]) {
      val modules = module.asInstanceOf[Graph[Float]].getForwardExecutions
      var i = 0
      modules.foreach(node => {
        if (node.nextNodes.length > 1) return
        if (!node.element.isInstanceOf[ReLU[Float]] && !node.element.isInstanceOf[Dropout[Float]]
          && !node.element.isInstanceOf[InferReshape[Float]]
          && !node.element.isInstanceOf[Reshape[Float]]
          && !node.element.isInstanceOf[View[Float]]
          && node.element.output != null) {
          if (i % 2 == 0) {
            node.element.output.toTensor[Float].set(out1)
          } else {
            node.element.output.toTensor[Float].set(out2)
          }
          i += 1
        }
      })
    }
  }

  /**
   * Load model weights and bias from source model to target model
   *
   * @param srcModel        source model
   * @param targetModel     target model
   * @param matchAll whether to match all layers' weights and bias,
   *                 if not, only load existing source weights and bias
   * @return
   */
  def loadModelWeights(srcModel: Module[Float], targetModel: Module[Float],
    matchAll: Boolean = true): this.type = {
    val srcParameter = srcModel.getParametersTable()
    val targetParameter = targetModel.getParametersTable()
    copyWeights(targetParameter, srcParameter, matchAll)
    this
  }

  private def copyWeights(target: Table, src: Table, matchAll: Boolean): Unit = {
    target.foreach {
      case (name: String, targetParams: Table) =>
        if (src.contains(name)) {
          val srcParams = src[Table](name)
          if (srcParams.contains("weight")) {
            val w = srcParams[Tensor[Float]]("weight")
            val tw = targetParams[Tensor[Float]]("weight")
            if (tw.size().sameElements(w.size())) {
              tw.copy(w)
            } else {
              logger.warn(s"$name weight size does not match, ignore ...")
            }
          }
          if (srcParams.contains("bias")) {
            val b = srcParams[Tensor[Float]]("bias")
            val tb = targetParams[Tensor[Float]]("bias")
            if (tb.size().sameElements(b.size())) {
              tb.copy(b)
            } else {
              logger.warn(s"$name bias size does not match, ignore ...")
            }
          }
        } else {
          if (matchAll) new Exception(s"module $name cannot find corresponding weight bias")
        }
    }
  }

  /**
   * Create Convolution layer followed by Relu activation
   * @param prevNodes previous node for convolution layer node
   * @param p convolution size info.
   *          Should be (input plane number, output plane number, kernel size, stride size,
   *          pad size).
   *          We'are assuming kernel width = kernel height, stride width = stride height,
   *          pad width = pad height
   * @param name layer name
   * @param prefix prefix for the layer name
   * @param nGroup kernel group number
   * @param propogateBack whether to propagate gradient back
   * @param ev
   * @tparam T
   * @return Relu node
   */
  def addConvRelu[@specialized(Float, Double) T: ClassTag](prevNodes: ModuleNode[T],
                                                           p: (Int, Int, Int, Int, Int),
                                                           name: String,
                                                           prefix: String = "conv",
                                                           nGroup: Int = 1,
                                                           propogateBack: Boolean = true)
                                                          (implicit ev: TensorNumeric[T])
  : ModuleNode[T] = {
    val conv = SpatialConvolution[T](p._1, p._2, p._3, p._3, p._4, p._4,
      p._5, p._5, nGroup = nGroup, propagateBack = propogateBack)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
      .setName(s"$prefix$name").inputs(prevNodes)
    ReLU[T](true).setName(s"relu$name").inputs(conv)
  }


  /**
   * Stop the input gradient of layers whose name ended with priorbox in a model,
   * their input gradient are not computed.
   *
   * @param model the graph model
   * @tparam T
   */
  def stopGradient[@specialized(Float, Double) T: ClassTag](model: Graph[T]): Unit = {
    val priorboxNames = model.modules
      .filter(_.getClass.getName.toLowerCase().endsWith("priorbox"))
      .map(_.getName()).toArray
    model.stopGradient(priorboxNames)
  }

  /**
    * Set Scale of gradientBias to the layers except specified layer,
    * their input gradient are not computed.
    *
    * @param model the graph model
    * @param b the value of the scale of gradientBias
    * @param exclude exclude layer name
    * @tparam T
    */
  def setScaleB[@specialized(Float, Double) T: ClassTag](model: Graph[T],
                                                         b: Double,
                                                         exclude: String): Unit = {
    val scaleModules = model.modules
      .filterNot(_.getName.endsWith(exclude))
      .toArray
    scaleModules.foreach(m => m.setScaleB(b))
  }

  /**
   * select results (confs || locs || priorboxes), use JoinTable to concat them into one tensor
   * @param start start index of the result
   * @param dim dimension to join
   * @param nInputDims specify the number of dimensions for the input
   * @param name result layer name
   * @return
   */
  def selectResults[@specialized(Float, Double) T: ClassTag](start: Int, dim: Int,
                                                             nInputDims: Int, numComponents: Int,
                                                             name: String)
                                                            (implicit ev: TensorNumeric[T]):
  Module[T] = {
    val con = ConcatTable[Activity, T]().setName(s"select results $name")
    var i = start
    while (i <= numComponents * 3) {
      con.add(SelectTable(i).setName(s"select $name $i"))
      i += 3
    }
    Sequential[T]().add(con).add(JoinTable[T](dim, nInputDims).setName(name))
  }

}

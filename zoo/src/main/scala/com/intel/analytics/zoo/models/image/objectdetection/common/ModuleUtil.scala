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
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table
import org.apache.log4j.Logger

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

}

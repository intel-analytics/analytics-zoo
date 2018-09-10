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

package com.intel.analytics.zoo.models.image.objectdetection.common.dataset

import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.models.image.objectdetection.common.BboxUtil

class FrcnnMiniBatch(val input: Table, val target: Tensor[Float])
  extends MiniBatch[Float] {

  private val targetIndices = if (target != null) BboxUtil.getGroundTruthIndices(target) else null

  override def size(): Int = input.length()

  override def slice(offset: Int, length: Int): MiniBatch[Float] = {
    require(length == 1, "only batch 1 is supported")
    val subInput = input[Table](offset)
    val subTarget = if (target != null) {
      var i = 0
      val targetOffset = targetIndices(offset - 1)._1
      var targetLength = 0
      while (i < length) {
        targetLength += targetIndices(offset + i - 1)._2
        i += 1
      }
      val subt = target.narrow(1, targetOffset, targetLength)
      subInput.update(3, subt)
      subt
    } else null
    FrcnnMiniBatch(subInput, subTarget)
  }

  override def getInput(): Activity = input

  override def getTarget(): Activity = target

  def getSample(): Table = getInput().toTable[Table](1)

  override def set(samples: Seq[Sample[Float]])(implicit ev: TensorNumeric[Float]): this.type = {
    throw new NotImplementedError("do not use Sample here")
  }
}

object FrcnnMiniBatch {
  def apply(input: Table, target: Tensor[Float]): FrcnnMiniBatch = {
    new FrcnnMiniBatch(input, target)
  }

  def getBboxes(batchInput: Tensor[Float]): Tensor[Float] = {
    batchInput.narrow(2, 4, 4)
  }

  val imageIndex = 1
  val labelIndex = 2
  val difficultIndex = 3
  val x1Index = 4
  val y1Index = 5
  val x2Index = 6
  val y2Index = 7
}


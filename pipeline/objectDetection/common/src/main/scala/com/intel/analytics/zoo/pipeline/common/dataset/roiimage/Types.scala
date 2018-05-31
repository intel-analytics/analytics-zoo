/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.zoo.pipeline.common.dataset.roiimage

import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil


/**
  * A batch of data feed into the model. The first size is batchsize
  *
  * @param input
  * @param target
  */
class SSDMiniBatch(val input: Tensor[Float], val target: Tensor[Float],
                   val imInfo: Tensor[Float] = null)
  extends MiniBatch[Float] {

  private val targetIndices = if (target != null) BboxUtil.getGroundTruthIndices(target) else null
  var imageFeatures: Array[ImageFeature] = _

  override def size(): Int = {
    input.size(1)
  }

  override def getInput(): Tensor[Float] = {
    input
  }

  override def getTarget(): Tensor[Float] = {
    target
  }

  override def slice(offset: Int, length: Int): MiniBatch[Float] = {
    val subInput = input.narrow(1, offset, length)
    val subTarget = if (target != null) {
      var i = 0
      val targetOffset = targetIndices(offset - 1)._1
      var targetLength = 0
      while (i < length) {
        targetLength += targetIndices(offset + i - 1)._2
        i += 1
      }
      target.narrow(1, targetOffset, targetLength)
    } else null

    SSDMiniBatch(subInput, subTarget)
  }

  override def set(samples: Seq[Sample[Float]])(implicit ev: TensorNumeric[Float])
  : SSDMiniBatch.this.type = {
    throw new NotImplementedError("do not use Sample here")
  }
}

object SSDMiniBatch {
  def apply(data: Tensor[Float], labels: Tensor[Float], imInfo: Tensor[Float] = null):
  SSDMiniBatch = new SSDMiniBatch(data, labels, imInfo)
}


case class ByteRecord(var data: Array[Byte], path: String)

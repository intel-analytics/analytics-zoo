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

package com.intel.analytics.zoo.models.image.objectdetection.common.nn

import com.intel.analytics.bigdl.nn.Linear
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.models.image.objectdetection.common.nn.BboxPred._
import org.apache.log4j.Logger

import scala.reflect.ClassTag

class BboxPred[T: ClassTag](inputSize: Int,
  outputSize: Int,
  withBias: Boolean = true,
  wRegularizer: Regularizer[T] = null,
  bRegularizer: Regularizer[T] = null,
  nClass: Int,
  var normalized: Boolean = false)(implicit ev: TensorNumeric[T]) extends Linear[T](
  inputSize, outputSize, withBias, wRegularizer, bRegularizer) {

  private val means = ProposalTarget.BBOX_NORMALIZE_MEANS.reshape(Array(1, 4))
    .expand(Array(nClass, 4)).reshape(Array(nClass * 4)).asInstanceOf[Tensor[T]]
  private val stds = ProposalTarget.BBOX_NORMALIZE_STDS.reshape(Array(1, 4))
    .expand(Array(nClass, 4)).reshape(Array(nClass * 4)).asInstanceOf[Tensor[T]]
  private val originalWeight = Tensor[T](1)
  private val originalBias = Tensor[T](1)


  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (!isTraining() && !normalized) {
      logger.info("normalize weight for test")
      originalWeight.resizeAs(weight).copy(weight)
      originalBias.resizeAs(bias).copy(bias)
      weight.cmul(stds.reshape(Array(nClass * 4, 1))
        .expand(Array(nClass * 4, weight.size(2))))
      bias.cmul(stds).add(ev.fromType(1), means)
      normalized = true
    } else if (isTraining() && normalized) {
      logger.info("restore net to original state for training")
      // restore net to original state
      weight.copy(originalWeight)
      bias.copy(originalBias)
      normalized = false
    }
    super.updateOutput(input)
  }
}

object BboxPred {
  val logger = Logger.getLogger(getClass)

  def apply[@specialized(Float, Double) T: ClassTag](inputSize: Int,
    outputSize: Int,
    withBias: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null, nClass: Int,
    normalized: Boolean = false)(implicit ev: TensorNumeric[T]): BboxPred[T] =
    new BboxPred[T](inputSize, outputSize, withBias, wRegularizer, bRegularizer, nClass, normalized)
}

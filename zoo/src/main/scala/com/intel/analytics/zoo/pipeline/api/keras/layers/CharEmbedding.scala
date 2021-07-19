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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential

import scala.reflect.ClassTag


/**
 * Turn sequence of character indices into dense word vectors of fixed size.
 * The input of this layer should be 2D.
 *
 * This layer can only be used as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * References
 *   - [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615)
 *
 * @param inputDim Int > 0. Size of the alphabet, ie. 1 + maximum integer
 *                 index occurring in the input data.
 *                 Each character index in the input should be within range [0, inputDim-1].
 * @param outputDim Int > 0. Dimension of the dense character-level word embedding.
 * @param charEmbedDim Int > 0. Dimension of the dense character embedding.
 * @param kernelRow Int > 0. Number of rows in the char-cnn kernel.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class CharEmbedding[T: ClassTag](
    val inputDim: Int,
    val outputDim: Int,
    val inputLength: Int,
    val kernelRow: Int,
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape)) with Net {

  require(inputDim > 0, s"inputDim of Embedding must be a positive integer, but got $inputDim")
  require(outputDim > 0, s"outputDim of Embedding must be a positive integer, but got $outputDim")
  require(kernelRow > 0, s"kernelRow must be a positive integer, but got $kernelRow")

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 2,
      s"CharEmbedding requires 2D input, but got input dim ${input.length}")
    Shape(input(0), outputDim)
  }

  def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val model = Sequential[T]()
    val layer = Embedding(
      inputDim = inputDim,
      outputDim = charEmbedDim,
      inputLength = input(1))
    model.add(layer)
    model.add(Convolution1D(
      nbFilter = outputDim,
      filterLength = kernelRow))
    model.add(GlobalMaxPooling1D())
    model.add(Highway())
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object CharEmbedding {
  def apply[@specialized(Float, Double) T: ClassTag](
      inputDim: Int,
      outputDim: Int,
      charEmbedDim: Int = 50,
      inputLength: Int,
      kernelRow: Int = 2)(implicit ev: TensorNumeric[T]): CharEmbedding[T] = {
    new CharEmbedding[T](inputDim, outputDim, charEmbedDim, kernelRow, Shape(inputLength))
  }
}

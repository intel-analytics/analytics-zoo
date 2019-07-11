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

package com.intel.analytics.zoo.pipeline.api.keras.layers.internal

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class InternalNarrow[T: ClassTag](dimension: Int, offset: Int, length: Int = 1,
  inplace: Boolean = false)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dim = if (dimension < 0) input.dim() + dimension + 1 else dimension
    val length = if (this.length < 0) input.size(dim) - offset + this.length + 2 else this.length
    val outputNarrow = input.narrow(dim, offset, length)
    if (inplace) {
      output = outputNarrow
    } else {
      output.resizeAs(outputNarrow).copy(outputNarrow)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val dim = if (dimension < 0) input.dim() + dimension + 1 else dimension
    val length = if (this.length < 0) input.size(dim) - offset + this.length + 2 else this.length
    gradInput.resizeAs(input).zero()
    // TODO: MAY NEED INPLACE UPDATE GRADINPUT
    gradInput.narrow(dim, offset, length).copy(gradOutput)
    gradInput
  }
  override def toString(): String = {
    s"${getPrintName}($dimension, $offset, $length)"
  }
}


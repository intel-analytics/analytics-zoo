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

package com.intel.analytics.zoo.pipeline.api.keras2.models

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.keras.KerasLayerSerializable
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializer
import com.intel.analytics.zoo.pipeline.api.autograd.Variable
import com.intel.analytics.zoo.pipeline.api.keras.models.{ Model => Keras1Model}

import scala.reflect.ClassTag

class Model [T: ClassTag](private val _inputs : Seq[ModuleNode[T]],
      private val _outputs : Seq[ModuleNode[T]])
      (implicit ev: TensorNumeric[T]) extends Keras1Model[T](_inputs, _outputs){
}

object Model extends KerasLayerSerializable {
  ModuleSerializer.registerModule(
    "com.intel.analytics.zoo.pipeline.api.keras2.models.Model",
    Model)

  /**
    * Build a multiple-input, multiple-output graph container.
    *
    * @param inputs  Array of input nodes.
    * @param outputs Array of output nodes.
    * @return A graph container.
    */
  def apply[T: ClassTag](
      inputs: Array[ModuleNode[T]],
      outputs: Array[ModuleNode[T]])(implicit ev: TensorNumeric[T]): Model[T] = {
    new Model[T](inputs, outputs)
  }

  /**
    * Build a single-input, multiple-output graph container
    *
    * @param inputs  The input node.
    * @param outputs Array of output nodes.
    * @return A graph container.
    */
  def apply[T: ClassTag](inputs: ModuleNode[T], outputs: Array[ModuleNode[T]])
      (implicit ev: TensorNumeric[T]): Model[T] = {
    new Model[T](Seq(inputs), outputs)
  }

  /**
    * Build a multiple-input, single-output graph container.
    *
    * @param inputs  Array of input nodes.
    * @param outputs The output node.
    * @return A graph container.
    */
  def apply[T: ClassTag](inputs: Array[ModuleNode[T]], outputs: ModuleNode[T])
      (implicit ev: TensorNumeric[T]): Model[T] = {
    new Model[T](inputs, Seq(outputs))
  }

  /**
    * Build a single-input, single-output graph container
    *
    * @param inputs  The input node.
    * @param outputs The output node.
    * @return A graph container.
    */
  def apply[T: ClassTag](inputs: ModuleNode[T], outputs: ModuleNode[T])
      (implicit ev: TensorNumeric[T]): Model[T] = {
    new Model[T](Seq(inputs), Seq(outputs))
  }

  /* ------------------------ factory methods for variables--------------------- */
  /**
    * Build a multiple-input, multiple-output graph container.
    *
    * @param inputs  Array of input variables.
    * @param outputs Array of output variables.
    * @return A graph container.
    */
  def apply[T: ClassTag](
      inputs: Array[Variable[T]],
      outputs: Array[Variable[T]])(implicit ev: TensorNumeric[T]): Model[T] = {
    new Model[T](inputs.map(_.node), outputs.map(_.node))
  }

  /**
    * Build a single-input, multiple-output graph container
    *
    * @param inputs  The input variable.
    * @param outputs Array of output variables.
    * @return A graph container.
    */
  def apply[T: ClassTag](inputs: Variable[T], outputs: Array[Variable[T]])
      (implicit ev: TensorNumeric[T]): Model[T] = {
    new Model[T](Seq(inputs.node), outputs.map(_.node))
  }

  /**
    * Build a multiple-input, single-output graph container.
    *
    * @param inputs  Array of input variables.
    * @param outputs The output variables.
    * @return A graph container.
    */
  def apply[T: ClassTag](inputs: Array[Variable[T]], outputs: Variable[T])
      (implicit ev: TensorNumeric[T]): Model[T] = {
    new Model[T](inputs.map(_.node), Seq(outputs.node))
  }

  /**
    * Build a single-input, single-output graph container
    *
    * @param inputs  The input variable.
    * @param outputs The output variable.
    * @return A graph container.
    */
  def apply[T: ClassTag](inputs: Variable[T], outputs: Variable[T])
      (implicit ev: TensorNumeric[T]): Model[T] = {
    new Model[T](Seq(inputs.node), Seq(outputs.node))
  }
}

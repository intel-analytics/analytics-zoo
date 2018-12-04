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

package com.intel.analytics.zoo.pipeline.api.keras2.layers


import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Input, Keras2Test, KerasBaseSpec}
import com.intel.analytics.zoo.pipeline.api.keras2.models.Model
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Minimum.minimum

import scala.util.Random


class MinimumSpec extends KerasBaseSpec {

  "Minimum" should "work properly" taggedAs(Keras2Test) in {
    val input1 = Tensor[Float](3, 8).rand(0, 1)
    val input2 = Tensor[Float](3, 8).rand(1, 2)
    val input = T(1 -> input1, 2 -> input2)
    val l1 = Input[Float](inputShape = Shape(8))
    val l2 = Input[Float](inputShape = Shape(8))
    val layer = Minimum[Float]().inputs(Array(l1, l2))
    val model = Model[Float](Array(l1, l2), layer)
    model.getOutputShape().toSingle().toArray should be (Array(-1, 8))
    model.forward(input) should be (input1)
  }

  "minimum" should "work properly" taggedAs(Keras2Test) in {
    val input1 = Tensor[Float](3, 8).rand(0, 1)
    val input2 = Tensor[Float](3, 8).rand(1, 2)
    val input = T(1 -> input1, 2 -> input2)
    val l1 = Input[Float](inputShape = Shape(8))
    val l2 = Input[Float](inputShape = Shape(8))
    val layer = minimum(inputs = List(l1, l2))
    val model = Model[Float](Array(l1, l2), layer)
    model.getOutputShape().toSingle().toArray should be (Array(-1, 8))
    model.forward(input) should be (input1)
  }

}

class MinimumSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val l1 = Input[Float](inputShape = Shape(8))
    val l2 = Input[Float](inputShape = Shape(8))
    val layer = Minimum[Float]().inputs(Array(l1, l2))
    val model = Model[Float](Array(l1, l2), layer)
    model.build(Shape(List(Shape(3, 8), Shape(3, 8))))
    val input1 = Tensor[Float](3, 8).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](3, 8).apply1(e => Random.nextFloat())
    val input = T(1 -> input1, 2 -> input2)
    runSerializationTest(model, input)
  }
}

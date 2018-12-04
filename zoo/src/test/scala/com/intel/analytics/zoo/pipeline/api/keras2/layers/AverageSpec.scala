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
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Input, InputLayer, Keras2Test, KerasBaseSpec}
import com.intel.analytics.zoo.pipeline.api.keras2.models.Model
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Average.average

import scala.util.Random


class AverageSpec extends KerasBaseSpec{
  "Average" should "work properly" taggedAs(Keras2Test) in {
    val input1 = Tensor[Float](3, 10).rand(0, 1)
    val input2 = Tensor[Float](3, 10).rand(1, 2)
    val input3 = Tensor[Float](3, 10).rand(2, 3)
    val input = T(1 -> input1, 2 -> input2, 3 -> input3)
    val l1 = Input[Float](inputShape = Shape(10))
    val l2 = Input[Float](inputShape = Shape(10))
    val l3 = Input[Float](inputShape = Shape(10))
    val layer = Average[Float]().inputs(Array(l1, l2, l3))
    val model = Model[Float](Array(l1, l2, l3), layer)
    model.getOutputShape().toSingle().toArray should be (Array(-1, 10))
    model.forward(input) should be ((input1 + input2 + input3)/3)
  }

  "average" should "work properly" taggedAs(Keras2Test) in {
    val input1 = Tensor[Float](3, 10).rand(0, 1)
    val input2 = Tensor[Float](3, 10).rand(1, 2)
    val input3 = Tensor[Float](3, 10).rand(2, 3)
    val input = T(1 -> input1, 2 -> input2, 3 -> input3)
    val l1 = Input[Float](inputShape = Shape(10))
    val l2 = Input[Float](inputShape = Shape(10))
    val l3 = Input[Float](inputShape = Shape(10))
    val layer = average(inputs = List(l1, l2, l3))
    val model = Model[Float](Array(l1, l2, l3), layer)
    model.getOutputShape().toSingle().toArray should be (Array(-1, 10))
    model.forward(input) should be ((input1 + input2 + input3)/3)
  }
}

class AverageSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val l1 = Input[Float](inputShape = Shape(10))
    val l2 = Input[Float](inputShape = Shape(10))
    val l3 = Input[Float](inputShape = Shape(10))
    val layer = Average[Float]().inputs(Array(l1, l2, l3))
    val model = Model[Float](Array(l1, l2, l3), layer)
    model.build(Shape(List(Shape(3, 10), Shape(3, 10), Shape(3, 10))))
    val input1 = Tensor[Float](3, 10).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](3, 10).apply1(e => Random.nextFloat())
    val input3 = Tensor[Float](3, 10).apply1(e => Random.nextFloat())
    val input = T(1 -> input1, 2 -> input2, 3 -> input3)
    runSerializationTest(model, input)
  }
}

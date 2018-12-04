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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.KerasBaseSpec
import com.intel.analytics.zoo.pipeline.api.keras2.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import com.intel.analytics.zoo.pipeline.api.keras.layers.Keras2Test

import scala.util.Random

class AveragePooling1DSpec extends KerasBaseSpec {

  "AveragePooling1D valid mode" should "be the same as Keras" taggedAs(Keras2Test) in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[12, 16])
        |input = np.random.random([3, 12, 16])
        |output_tensor = AveragePooling1D()(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = AveragePooling1D[Float](inputShape = Shape(12, 16))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 6, 16))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "AveragePooling1D same mode" should "be the same as Keras" taggedAs(Keras2Test) in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[32, 32])
        |input = np.random.random([2, 32, 32])
        |output_tensor = AveragePooling1D(pool_size=3, strides=1, padding="same")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = AveragePooling1D[Float](poolSize = 3, strides = 1, padding = "same",
      inputShape = Shape(32, 32))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 32, 32))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class AveragePooling1DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = AveragePooling1D[Float](inputShape = Shape(12, 16))
    layer.build(Shape(2, 12, 16))
    val input = Tensor[Float](2, 12, 16).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}

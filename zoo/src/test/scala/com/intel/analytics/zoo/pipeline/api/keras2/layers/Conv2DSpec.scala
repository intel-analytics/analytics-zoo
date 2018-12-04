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
import com.intel.analytics.zoo.pipeline.api.keras2.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Keras2Test, KerasBaseSpec, Permute}

import scala.util.Random

class Conv2DSpec extends KerasBaseSpec {
  def weightConverterNCHW(in: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    in.length match {
      case 1 =>
        in(0).resize(Array(1) ++ in(0).size())
        val shape: Array[Int] = in(0).size()
        val weights = Permute[Float](Array(4, 3, 1, 2), inputShape = Shape(shape))
          .doBuild(inputShape = Shape(shape)).forward(in(0))
        Array(weights)
      case _ =>
        in(0).resize(Array(1) ++ in(0).size())
        val shape: Array[Int] = in(0).size()
        val weights = Permute[Float](Array(4, 3, 1, 2), inputShape = Shape(shape))
          .doBuild(inputShape = Shape(shape)).forward(in(0))
        Array(weights, in(1))
    }
  }
  def weightConverterNHWC(in: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    in.length match {
      case 1 => in
      case _ =>
        Array(in(0).resize(Array(1) ++ in(0).size()), in(1))
    }
  }

  "Conv2D NCHW" should "be the same as Keras" taggedAs(Keras2Test) in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 24, 24])
        |input = np.random.random([2, 3, 24, 24])
        |output_tensor = Conv2D(64, (2, 5), activation="relu",
        |                              data_format="channels_first")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = Conv2D[Float](64, Array(2, 5), activation = "relu",
      inputShape = Shape(3, 24, 24))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverterNCHW, 1e-3)
  }

  "Conv2D NHWC" should "be the same as Keras" taggedAs(Keras2Test) in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[24, 24, 3])
        |input = np.random.random([2, 24, 24, 3])
        |output_tensor = Conv2D(32, (4, 4), padding="same",
        |                              data_format="channels_last")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = Conv2D[Float](32, Array(4, 4), dataFormat = "channels_last",
      padding = "same", inputShape = Shape(24, 24, 3))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverterNHWC, 1e-3)
  }

  "Conv2D without bias" should "be the same as Keras" taggedAs(Keras2Test) in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 24, 24])
        |input = np.random.random([2, 3, 24, 24])
        |output_tensor = Conv2D(64, kernel_size=(2, 5), use_bias=False, strides=(2, 3),
        |                              kernel_initializer="normal",
        |                              data_format="channels_first")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = Conv2D[Float](64, Array(2, 5), strides = Array(2, 3),
      kernelInitializer = "normal", useBias = false, inputShape = Shape(3, 24, 24))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverterNCHW, 1e-4)
  }
}

class Conv2DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Conv2D[Float](64, Array(2, 5), inputShape =
      Shape(3, 24, 24))
    layer.build(Shape(2, 3, 24, 24))
    val input = Tensor[Float](2, 3, 24, 24).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}

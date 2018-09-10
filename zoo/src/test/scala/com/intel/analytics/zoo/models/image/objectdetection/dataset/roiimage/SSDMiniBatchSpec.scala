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

package com.intel.analytics.zoo.models.image.objectdetection.dataset.roiimage

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.roiimage.SSDMiniBatch
import org.scalatest.{FlatSpec, Matchers}

class SSDMiniBatchSpec extends FlatSpec with Matchers {
  "SSDMiniBatch slice" should "work properly" in {
    val data = Tensor[Float](5, 5)
    val target = Tensor[Float](20, 5)

    var i = 0
    while (i < 5) {
      data(i + 1).fill(i)
      target(i * 4 + 1).fill(i)
      target(i * 4 + 2).fill(i)
      target(i * 4 + 3).fill(i)
      target(i * 4 + 4).fill(i)
      i += 1
    }

    val minibatch = SSDMiniBatch(data, target)

    val s1 = minibatch.slice(1, 1)

    val i1 = s1.getInput().toTensor[Float]
    i1.size(1) should be(1)
    i1.apply1(x => {
      assert(x == 0)
      x
    })

    val t1 = s1.getTarget().toTensor[Float]
    t1.size(1) should be (4)
    t1.apply1(x => {
      assert(x == 0)
      x
    })

    val s2 = minibatch.slice(2, 2)

    val i2 = s2.getInput().toTensor[Float]
    i2.size(1) should be(2)
    i2(1).apply1(x => {
      assert(x == 1)
      x
    })
    i2(2).apply1(x => {
      assert(x == 2)
      x
    })

    val t2 = s2.getTarget().toTensor[Float]
    t2.size(1) should be (8)
    println(t2)
    t2.narrow(1, 1, 4).apply1(x => {
      assert(x == 1)
      x
    })
    t2.narrow(1, 5, 4).apply1(x => {
      assert(x == 2)
      x
    })

    val s3 = minibatch.slice(5, 1)

    val i3 = s3.getInput().toTensor[Float]
    i3.size(1) should be(1)
    i3(1).apply1(x => {
      assert(x == 4)
      x
    })
    println(i3)

    val t3 = s3.getTarget().toTensor[Float]
    t3.size(1) should be (4)
    t3.narrow(1, 1, 4).apply1(x => {
      assert(x == 4)
      x
    })
  }

  "SSDMiniBatch slice2" should "work properly" in {
    val data = Tensor[Float](5, 5)
    val target = Tensor[Float](20, 5)

    var i = 0
    while (i < 5) {
      data(i + 1).fill(i)
      target(i * 4 + 1).fill(i)
      target(i * 4 + 2).fill(i)
      target(i * 4 + 3).fill(i)
      target(i * 4 + 4).fill(i)
      i += 1
    }

    val minibatch = SSDMiniBatch(data, target)

    val s1 = minibatch.slice(1, 1)

    val i1 = s1.getInput().toTensor[Float]
    i1.size(1) should be(1)
    i1.apply1(x => {
      assert(x == 0)
      x
    })

    val t1 = s1.getTarget().toTensor[Float]
    t1.size(1) should be (4)
    t1.apply1(x => {
      assert(x == 0)
      x
    })

    val s2 = minibatch.slice(2, 2)

    val i2 = s2.getInput().toTensor[Float]
    i2.size(1) should be(2)
    i2(1).apply1(x => {
      assert(x == 1)
      x
    })
    i2(2).apply1(x => {
      assert(x == 2)
      x
    })

    val t2 = s2.getTarget().toTensor[Float]
    t2.size(1) should be (8)
    t2.narrow(1, 1, 4).apply1(x => {
      assert(x == 1)
      x
    })
    t2.narrow(1, 5, 4).apply1(x => {
      assert(x == 2)
      x
    })

    val s3 = minibatch.slice(5, 1)

    val i3 = s3.getInput().toTensor[Float]
    i3.size(1) should be(1)
    i3(1).apply1(x => {
      assert(x == 4)
      x
    })

    val t3 = s3.getTarget().toTensor[Float]
    t3.size(1) should be (4)
    t3.narrow(1, 1, 4).apply1(x => {
      assert(x == 4)
      x
    })
  }

  "SSDMiniBatch slice with empty target" should "work properly" in {
    val data = Tensor[Float](5, 5)
    val target = Tensor[Float](20, 5)

    var i = 0
    while (i < 5) {
      data(i + 1).fill(i)
      target(i * 4 + 1).fill(i)
      target(i * 4 + 2).fill(i)
      target(i * 4 + 3).fill(i)
      target(i * 4 + 4).fill(i)
      i += 1
    }

    val minibatch = SSDMiniBatch(data, target)

    val s1 = minibatch.slice(1, 1)

    val i1 = s1.getInput().toTensor[Float]
    i1.size(1) should be(1)
    i1.apply1(x => {
      assert(x == 0)
      x
    })

    val t1 = s1.getTarget().toTensor[Float]
    t1.size(1) should be (4)
    t1.apply1(x => {
      assert(x == 0)
      x
    })

    val s2 = minibatch.slice(2, 2)

    val i2 = s2.getInput().toTensor[Float]
    i2.size(1) should be(2)
    i2(1).apply1(x => {
      assert(x == 1)
      x
    })
    i2(2).apply1(x => {
      assert(x == 2)
      x
    })

    val t2 = s2.getTarget().toTensor[Float]
    t2.size(1) should be (8)
    t2.narrow(1, 1, 4).apply1(x => {
      assert(x == 1)
      x
    })
    t2.narrow(1, 5, 4).apply1(x => {
      assert(x == 2)
      x
    })

    val s3 = minibatch.slice(5, 1)

    val i3 = s3.getInput().toTensor[Float]
    i3.size(1) should be(1)
    i3(1).apply1(x => {
      assert(x == 4)
      x
    })

    val t3 = s3.getTarget().toTensor[Float]
    t3.size(1) should be (4)
    t3.narrow(1, 1, 4).apply1(x => {
      assert(x == 4)
      x
    })
  }

  "SSDMinibatch slice " should "work properly" in {
    val data = Tensor[Float](7, 5)

    val label = Tensor(Storage(Array(
      0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
      1.0, 20.0, 0.0, 0.0603052, 0.368519, 0.848145, 1.0,
      1.0, 16.0, 0.0, 0.89412, 0.767627, 0.98189, 1.0,
      2.0, 16.0, 0.0, 0.645249, 0.577347, 0.731589, 0.808865,
      2.0, 15.0, 0.0, 0.614338, 0.646141, 0.850972, 0.83797,
      3.0, 8.0, 0.0, 0.241746, 0.322738, 0.447184, 0.478388,
      3.0, 8.0, 0.0, 0.318659, 0.336546, 0.661729, 0.675461,
      3.0, 8.0, 0.0, 0.56154, 0.300144, 0.699173, 0.708098,
      3.0, 8.0, 0.0, 0.220494, 0.327759, 0.327767, 0.396797,
      3.0, 8.0, 0.0, 0.194182, 0.317717, 0.279191, 0.389266,
      4.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
      5.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
      6.0, 10.0, 0.0, 0.67894, 0.471823, 0.929308, 0.632044,
      6.0, 10.0, 0.0, 0.381443, 0.572376, 0.892489, 0.691713,
      7.0, 9.0, 0.0, 0.0, 0.0620616, 0.667269, 1.0
    ).map(_.toFloat))).resize(15, 7)

    val minibatch = SSDMiniBatch(data, label)

    val s1 = minibatch.slice(1, 1)

    assert(s1.getTarget().toTensor[Float].valueAt(1, 1) == 0)
    assert(s1.getTarget().toTensor[Float].valueAt(1, 2) == -1)

    val s2 = minibatch.slice(2, 4)
    s2.getTarget().toTensor[Float].size(1) should be (10)
    s2.getTarget().toTensor[Float].valueAt(10, 1) should be(4)

  }
}


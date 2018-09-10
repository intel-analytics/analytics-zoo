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

package com.intel.analytics.zoo.models.image.objectdetection

import com.intel.analytics.zoo.models.image.objectdetection.common.EvalUtil
import org.scalatest.{FlatSpec, Matchers}


class EvalUtilSpec extends FlatSpec with Matchers {

  "computeAP" should "work properly" in {
    val detections = Array(
      (1.0f, 0, 1),
      (1.0f, 1, 0),
      (0.9f, 1, 0),
      (0.9f, 0, 1),
      (0.8f, 1, 0),
      (0.7f, 0, 1),
      (0.7f, 1, 0),
      (0.6f, 0, 1),
      (0.5f, 0, 1),
      (0.4f, 0, 1),
      (0.4f, 1, 0))

    val ap = EvalUtil.computeAP(detections, true, 5)

    assert((ap - 0.598662).abs < 1e-5)

  }

  "computeAP 2" should "work properly" in {
    val data = Array(
      0.0107105, 0, 1,
      0.0102218, 0, 1,
      0.0157222, 0, 1,
      0.0245589, 1, 0,
      0.0131284, 0, 1,
      0.995492, 1, 0,
      0.406693, 0, 1,
      0.0607406, 0, 1,
      0.0537384, 0, 1,
      0.0348978, 0, 1,
      0.028296, 0, 1,
      0.0256517, 0, 1,
      0.0240763, 0, 1,
      0.0174207, 0, 1,
      0.0170641, 0, 1,
      0.0160791, 0, 1,
      0.0159111, 0, 1,
      0.013588, 0, 1,
      0.0106639, 0, 1,
      0.0147799, 0, 1,
      0.0959367, 0, 1)

    val res = (data.indices by 3).map(i => {
      (data(i).toFloat, data(i + 1).toInt, data(i + 2).toInt)
    }).toArray

    val ap = EvalUtil.computeAP(res, true, 3)

    assert((ap - 0.424242).abs < 1e-6)
  }

  "ap 1" should "work properly" in {
    val data = Array(0.0112148, 0, 1,
      0.986149, 1, 0,
      0.0401057, 0, 1,
      0.999805, 1, 0,
      0.080021, 0, 1,
      0.0164427, 0, 1,
      0.0163025, 0, 1,
      0.999002, 1, 0,
      0.0224141, 0, 1,
      0.013316, 0, 1,
      0.0100836, 0, 1,
      0.757038, 0, 1,
      0.27257, 0, 1,
      0.0635232, 0, 1,
      0.0228072, 0, 1,
      0.0111696, 0, 1)

    val res = (data.indices by 3).map(i => {
      (data(i).toFloat, data(i + 1).toInt, data(i + 2).toInt)
    }).toArray

    val ap = EvalUtil.computeAP(res, true, 3)

    assert((ap - 1).abs < 1e-6)
  }
}

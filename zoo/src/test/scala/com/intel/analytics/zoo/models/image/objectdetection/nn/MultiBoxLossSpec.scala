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

package com.intel.analytics.zoo.models.image.objectdetection.nn

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.models.image.objectdetection.common.nn.{MultiBoxLoss, MultiBoxLossParam}
import org.scalatest.{FlatSpec, Matchers}

class MultiBoxLossSpec extends FlatSpec with Matchers {
  "MatchBbox" should "work properly" in {
    val param = MultiBoxLossParam(overlapThreshold = 0.3)
    val multiBoxLoss = new MultiBoxLoss[Float](param)
    val gtBbox = Tensor(Storage(Array(1, 1, 1, 0.1, 0.1, 0.3, 0.3,
      1, 1, 1, 0.3, 0.3, 0.6, 0.5
    ).map(x => x.toFloat))).resize(2, 7)

    val preBboxes = Tensor(Storage(Array(
      0.1, 0, 0.4, 0.3,
      0, 0.1, 0.2, 0.3,
      0.2, 0.1, 0.4, 0.4,
      0.4, 0.3, 0.7, 0.5,
      0.5, 0.4, 0.7, 0.7,
      0.7, 0.7, 0.8, 0.8
    ).map(x => x.toFloat))).resize(6, 4)

    val out = multiBoxLoss.matchBbox(gtBbox, preBboxes)

    out._1.length should be(6)
    out._2.length should be(6)

    out._1(0) should be(0)
    out._1(1) should be(0)
    out._1(2) should be(-1)
    out._1(3) should be(1)
    out._1(4) should be(-1)
    out._1(5) should be(-1)

    assert(Math.abs(out._2(0) - 4.0 / 9) < 1e-6)
    assert(Math.abs(out._2(1) - 2.0 / 6) < 1e-6)
    assert(Math.abs(out._2(2) - 2.0 / 8) < 1e-6)
    assert(Math.abs(out._2(3) - 4.0 / 8) < 1e-6)
    assert(Math.abs(out._2(4) - 1.0 / 11) < 1e-6)
    assert(Math.abs(out._2(5) - 0) < 1e-6)
  }

  "MultiBoxLoss" should "work properly" in {
    val mboxLoc = Tensor(Storage(Array(0.52501357, 0.37481785, 0.65692115, 0.9026065, 0.8785092,
      0.047891248, 0.2822641, 0.8266261, 0.30174193, 0.24188538,
      0.49466583, 0.26651323, 0.882372, 0.44199145, 0.081662744,
      0.93523306, 0.63634783, 0.87412876, 0.072484754, 0.27974287).map(_.toFloat))).resize(1, 20)

    val mboxConf = Tensor(Storage(Array(0.3262407, 0.6703484, 0.035577536, 0.9437432, 0.16391742,
      0.9050244, 0.46468765, 0.872331, 0.025942808, 0.86511457,
      0.35659435, 0.48196694, 0.52042866, 0.22129858, 0.67268646,
      0.10894631, 0.89542687, 0.98964524, 0.7730079, 0.040064808,
      0.48444122, 0.16309963, 0.4160194, 0.012772692, 0.26705426,
      0.6398565, 0.38015434, 0.8807492, 0.29181308, 0.642916,
      0.10633149, 0.71842605, 0.6209266, 0.9715023, 0.79098934,
      0.80236685, 0.99470294, 0.44911724, 0.036196362, 0.20092686,
      0.48896027, 0.7530397, 0.9909588, 0.43188113, 0.54178655,
      0.5485328, 0.62169033, 0.4306348, 0.39199877, 0.94761413,
      0.44464448, 0.20284249, 0.5731559, 0.06291938, 0.88646275,
      0.9530642, 0.70435226, 0.055710718, 0.93522036, 0.19499905,
      0.36204246, 0.59636503, 0.19602156, 0.510819, 0.41094393,
      0.39636, 0.36272812, 0.17927776, 0.9882963, 0.7319073,
      0.21066171, 0.53639853, 0.201604, 0.13473003, 0.1351613,
      0.21836676, 0.7223917, 0.48406565, 0.25093213, 0.82331216,
      0.4068227, 0.94689626, 0.62982035, 0.91242325, 0.6054889,
      0.8912483, 0.2734644, 0.30938464, 0.4141131, 0.7900512,
      0.99153, 0.93852603, 0.60857415, 0.17720197, 0.13556595,
      0.2102032, 0.5676692, 0.18344863, 0.10256101, 0.84535015,
      0.44111377, 0.5383418, 0.42354733, 0.19407134, 0.3391656,
      0.76854664, 0.47705686, 0.49052563, 0.29531467, 0.8412546).map(_.toFloat))).resize(1, 110)

    val mboxPriors = Tensor(Storage(Array(0.9863004, 0.8906429, 0.6163014, 0.5632963, 0.7019916,
      0.9861657, 0.615533, 0.24482808, 0.22638789, 0.31538585,
      0.7987784, 0.83047765, 0.9373293, 0.58650905, 0.666434,
      0.86567706, 0.49175686, 0.42000195, 0.39671823, 0.70550215,
      0.49680862, 0.562595, 0.72703135, 0.90935993, 0.46714246,
      0.60894376, 0.46564788, 0.57466185, 0.027301019, 0.12387657,
      0.6710133, 0.40470868, 0.57172865, 0.1711163, 0.410948,
      0.36043823, 0.5005803, 0.1234323, 0.9833935, 0.9885782).map(_.toFloat))).resize(1, 2, 20)

    val target = Tensor(Storage(Array(0.0, 11.0, 0.0, 0.337411, 0.468211, 0.429096, 0.516061)
      .map(_.toFloat))).resize(1, 7)
    val param = MultiBoxLossParam()
    val multiBoxLoss = new MultiBoxLoss[Float](param)
    val input = T()
    input.insert(mboxLoc)
    input.insert(mboxConf)
    input.insert(mboxPriors)


    val loss = multiBoxLoss.forward(input, target)

    loss should equal(30.262516f)
  }
}

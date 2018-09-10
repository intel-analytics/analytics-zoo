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

package com.intel.analytics.zoo.models.image.objectdetection.dataset

import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.Imdb
import org.scalatest.{FlatSpec, Matchers}

class PascalVocSpec extends FlatSpec with Matchers {
  val resource = getClass().getClassLoader().getResource("VOCdevkit")
  "pascal voc load images and annotations" should "work properly" in {
    val voc = Imdb.getImdb("voc_2007_testcode", resource.getPath)
    val roidb = voc.getRoidb().array
    roidb.length should be(2)
    roidb(1).uri() should be(resource.getPath + "/VOC2007/JPEGImages/000055.jpg")

    val expectedClasses =
      "15.0\t16.0\t16.0\t16.0\t\n0.0\t0.0\t1.0\t0.0".split("\t").map(x => x.trim.toFloat)

    val expectedBoxes = ("209.0\t132.0\t275.0\t222.0\t219.0\t107.0\t273.0\t187.0\t" +
      "465.0\t273.0\t483.0\t324.0\t436.0\t279.0\t456.0\t327.0\t")
        .split("\t").map(x => x.trim.toFloat)

    roidb(1).getLabel[RoiLabel].size() should be(4)
    roidb(1).getLabel[RoiLabel].classes.storage().array() should equal(expectedClasses)
    roidb(1).getLabel[RoiLabel].bboxes.storage().array() should equal(expectedBoxes)
  }
}

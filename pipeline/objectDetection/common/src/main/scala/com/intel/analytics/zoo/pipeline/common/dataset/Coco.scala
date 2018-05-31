/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.zoo.pipeline.common.dataset

import java.io.File
import java.nio.file.Paths

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame, LocalImageFrame}
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import org.apache.commons.io.FileUtils
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.parsing.json.JSON

class Coco(val imageSet: String, devkitPath: String) extends Imdb {

  override def getRoidb(readImage: Boolean = true): LocalImageFrame = {
    val imageSetFile = Paths.get(devkitPath, "ImageSets", s"$imageSet.txt").toFile
    assert(imageSetFile.exists(), "Path does not exist " + imageSetFile.getAbsolutePath)

    var imageSet_new = imageSet
    if (imageSet == "val2014" || imageSet == "minival2014" || imageSet == "valminusminival2014"){
      imageSet_new = "val2014"
    } else if (imageSet == "test-dev2015"){
      imageSet_new = "test2015"
    }
    val array = Source.fromFile(imageSetFile).getLines()
      .map(line => line.trim.split("\\s")).map(x => {
      val imagePath = devkitPath + "/images/" + imageSet_new + "/" + x(0) + ".jpg"
      val annoPath = devkitPath + "/Annotations/" + imageSet + "/" + x(0) + ".json"

      val image = if (readImage) loadImage(imagePath) else null
      ImageFeature(image, Coco.loadAnnotation(annoPath), imagePath)
    }).toArray

    ImageFrame.array(array)
  }
}

object Coco {

  val logger = Logger.getLogger(getClass.getName)

  // original coco category id and corresponding class names
  val cocoCatIdAndClassName = Array(
    (0, "__background__"),
    (1, "person"),
    (2, "bicycle"),
    (3, "car"),
    (4, "motorcycle"),
    (5, "airplane"),
    (6, "bus"),
    (7, "train"),
    (8, "truck"),
    (9, "boat"),
    (10, "traffic light"),
    (11, "fire hydrant"),
    (13, "stop sign"),
    (14, "parking meter"),
    (15, "bench"),
    (16, "bird"),
    (17, "cat"),
    (18, "dog"),
    (19, "horse"),
    (20, "sheep"),
    (21, "cow"),
    (22, "elephant"),
    (23, "bear"),
    (24, "zebra"),
    (25, "giraffe"),
    (27, "backpack"),
    (28, "umbrella"),
    (31, "handbag"),
    (32, "tie"),
    (33, "suitcase"),
    (34, "frisbee"),
    (35, "skis"),
    (36, "snowboard"),
    (37, "sports ball"),
    (38, "kite"),
    (39, "baseball bat"),
    (40, "baseball glove"),
    (41, "skateboard"),
    (42, "surfboard"),
    (43, "tennis racket"),
    (44, "bottle"),
    (46, "wine glass"),
    (47, "cup"),
    (48, "fork"),
    (49, "knife"),
    (50, "spoon"),
    (51, "bowl"),
    (52, "banana"),
    (53, "apple"),
    (54, "sandwich"),
    (55, "orange"),
    (56, "broccoli"),
    (57, "carrot"),
    (58, "hot dog"),
    (59, "pizza"),
    (60, "donut"),
    (61, "cake"),
    (62, "chair"),
    (63, "couch"),
    (64, "potted plant"),
    (65, "bed"),
    (67, "dining table"),
    (70, "toilet"),
    (72, "tv"),
    (73, "laptop"),
    (74, "mouse"),
    (75, "remote"),
    (76, "keyboard"),
    (77, "cell phone"),
    (78, "microwave"),
    (79, "oven"),
    (80, "toaster"),
    (81, "sink"),
    (82, "refrigerator"),
    (84, "book"),
    (85, "clock"),
    (86, "vase"),
    (87, "scissors"),
    (88, "teddy bear"),
    (89, "hair drier"),
    (90, "toothbrush"))

  val classes: Array[String] = cocoCatIdAndClassName.map(_._2)

  val cocoCatIdToClassInd: Map[Int, Int] = cocoCatIdAndClassName.zip(Stream.from(1)).map(x => {
    (x._1._1, x._2)
  }).toMap

  def loadAnnotation(path: String): RoiLabel = {
    val text = FileUtils.readFileToString(new File(path))
    val result = JSON.parseFull(text)
    result match {
      case Some(map: Map[String, Any]) =>
        val image = map("image").asInstanceOf[Map[String, Any]]
        val annotations = map("annotation").asInstanceOf[List[Map[String, Any]]]
        val width = image("width").asInstanceOf[Double]
        val height = image("height").asInstanceOf[Double]
        val validBoxes = new ArrayBuffer[Float]()
        val validClasses = new ArrayBuffer[Float]()

        var i = 0
        while (i < annotations.length) {
          val area = annotations(i)("area").asInstanceOf[Double]
          val boxes = annotations(i)("bbox").asInstanceOf[List[Double]]
          val x1 = Math.max(0, boxes(0)).toFloat
          val y1 = Math.max(0, boxes(1)).toFloat
          val x2 = Math.min(width - 1, x1 + Math.max(0, boxes(2) - 1)).toFloat
          val y2 = Math.min(height - 1, y1 + Math.max(0, boxes(3) - 1)).toFloat
          if (area > 0 && x2 >= x1 && y2 >= y1) {
            validBoxes.append(x1)
            validBoxes.append(y1)
            validBoxes.append(x2)
            validBoxes.append(y2)
            val clsInd = cocoCatIdToClassInd(
              annotations(i)("category_id").asInstanceOf[Double].toInt)
            validClasses.append(clsInd.toFloat)
          }
          i += 1
        }
        // compatible with pascal storage, difficults are all 0
        val clses = new Array[Float](validClasses.length * 2)
        validClasses.copyToArray(clses, 0, validClasses.length)
        RoiLabel(Tensor(Storage(clses)).resize(2, validClasses.length),
          Tensor(Storage(validBoxes.toArray)).resize(validBoxes.length / 4, 4))
      case _ => throw new Exception("Parse annotation failed.")
    }
  }
}


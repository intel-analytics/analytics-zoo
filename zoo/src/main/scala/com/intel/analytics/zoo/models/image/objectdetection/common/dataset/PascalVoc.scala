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

package com.intel.analytics.zoo.models.image.objectdetection.common.dataset

import java.io.File
import java.nio.file.Paths

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.feature.image.{ImageSet, LocalImageSet}

import scala.Array._
import scala.io.Source
import scala.xml.XML

/**
 * Parse the pascal voc dataset, load images and annotations
 * @param year the year of dataset
 * @param imageSet train, val, test, etc
 * @param devkitPath dataset folder
 */
class PascalVoc(val year: String = "2007", val imageSet: String,
  devkitPath: String) extends Imdb {
  val name = "voc_" + year + "_" + imageSet

  assert(new File(devkitPath).exists(),
    "VOCdevkit path does not exist: " + devkitPath)

  /**
   * Construct an image path from the image"s "index" identifier.
   * @param index e.g. 000001
   * @return image path
   */
  def imagePathFromIndex(index: String): String = s"JPEGImages/$index.jpg"

  def annotationPath(index: String): String = "Annotations/" + index + ".xml"

  override def getRoidb(readImage: Boolean = true): LocalImageSet = {
    val list = if (year == "0712") Array("2007", "2012") else Array(year)
    var imdexToPaths = Map[String, (String, String)]()
    list.foreach(y => {
      val dataPath = Paths.get(devkitPath, "/VOC" + y).toFile
      assert(dataPath.exists(), s"cannot find data folder ${dataPath} :for ${ name }")

      val imageSetFile = Paths.get(dataPath.toString, s"/ImageSets/Main/$imageSet.txt").toFile
      assert(imageSetFile.exists(), "Path does not exist " + imageSetFile.getAbsolutePath)
      Source.fromFile(imageSetFile).getLines().foreach(line => {
        val index = line.trim
        imdexToPaths += (index -> (dataPath + "/" + imagePathFromIndex(index),
          dataPath + "/" + annotationPath(index)))
      })
    })
    val array = imdexToPaths.toIterator.map(x => {
      val image = if (readImage) loadImage(x._2._1) else null
      ImageFeature(image,
        PascalVoc.loadAnnotation(x._2._2, PascalVoc.classToInd), x._2._1)
    }).toArray
    ImageSet.array(array)
  }
}

object PascalVoc {
  val classes = Array[String](
    "__background__", // always index 1 (1-based index)
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
  )

  val classToInd = (classes zip (Stream from 1).map(_.toFloat)).toMap

  def loadAnnotation(path: String, labelMap: Map[String, Float]): RoiLabel = {
    val xml = XML.loadFile(path)
    val objs = xml \\ "object"

    val boxes = Tensor[Float](objs.length, 4)
    val boxesArr = boxes.storage().array()
    val classNames = new Array[String](objs.length)
    val difficults = new Array[Float](objs.length)
    // Load object bounding boxes into a data frame.
    var ix = 0
    while (ix < objs.length) {
      val obj = objs(ix)
      val bndbox = obj \ "bndbox"
      boxesArr(ix * 4) = (bndbox \ "xmin").text.toFloat
      boxesArr(ix * 4 + 1) = (bndbox \ "ymin").text.toFloat
      boxesArr(ix * 4 + 2) = (bndbox \ "xmax").text.toFloat
      boxesArr(ix * 4 + 3) = (bndbox \ "ymax").text.toFloat
      classNames(ix) = (obj \ "name").text
      difficults(ix) = (obj \ "difficult").text.toFloat
      ix += 1
    }
    val classes = classNames.map(labelMap)
    val gtClasses = Tensor(Storage(classes ++ difficults)).resize(2, classes.length)
    RoiLabel(gtClasses, boxes)
  }
}

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

package com.intel.analytics.zoo.models.image.objectdetection.common

import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.PascalVoc._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import org.apache.commons.lang3.SerializationUtils
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer


/**
 *
 * @param imageSet voc_2007_test, voc_2007_train, etc
 */
class PascalVocEvaluator(imageSet: String) extends DetectionEvaluator {

  private val year: String = imageSet.split("_")(1)

  // The PASCAL VOC metric changed in 2010
  def use07metric: Boolean = year == "2007"


  def map(results: Array[(Int, Array[(Float, Int, Int)])]): Array[(String, Double)] = {
    import PascalVocEvaluator._
    val aps = ArrayBuffer[Double]()
    logger.info("VOC07 metric ? " + (if (use07metric) "yes" else "No"))
    val output = ArrayBuffer[(String, Double)]()
    var i = 0
    while (i < classes.length) {
      val cls = classes(i)
      if (cls != "__background__") {
        val ap = EvalUtil.computeAP(results(i)._2, use07Metric = use07metric, results(i)._1)
        aps.append(ap)
        logger.info(s"AP for $cls = ${ "%.4f".format(ap) }")
        output.append((cls, ap))
      }
      i += 1
    }
    printInfo(aps, output)
    output.toArray
  }


  def evaluateBatch(results: Array[Array[RoiLabel]], gt: Tensor[Float])
  : Array[(Int, Array[(Float, Int, Int)])] = {
    var i = 0
    val output = new Array[(Int, Array[(Float, Int, Int)])](classes.length)
    val gtLength = if (gt.nElement() == 0) 0 else gt.size(1)
    val gtAreas = Tensor[Float](gtLength)
    BboxUtil.getAreas(gt, gtAreas, 4)
    while (i < classes.length) {
      val cls = classes(i)
      if (cls != "__background__") {
        output(i) = EvalUtil.evaluateBatch(results, gt, gtAreas, i,
          ovThresh = 0.5, use07Metric = use07metric)
      } else {
        output(i) = (0, Array[(Float, Int, Int)]())
      }
      i += 1
    }
    output
  }

  def cloneEvaluator(): PascalVocEvaluator = {
    SerializationUtils.clone(this)
  }
}

object PascalVocEvaluator {
  val logger = Logger.getLogger(getClass)

  def meanAveragePrecision(results: Array[(Int, Array[(Float, Int, Int)])],
    use07metric: Boolean, classes: Array[String]): ArrayBuffer[(String, Float)] = {
    logger.info("VOC07 metric ? " + (if (use07metric) "yes" else "No"))
    val output = ArrayBuffer[(String, Float)]()
    var i = 0
    while (i < classes.length) {
      val cls = classes(i)
      if (cls != "__background__") {
        val ap = EvalUtil.computeAP(results(i)._2, use07Metric = use07metric, results(i)._1).toFloat
        output.append((cls, ap))
      }
      i += 1
    }
    output
  }

  private def printInfo(aps: ArrayBuffer[Double], output: ArrayBuffer[(String, Double)]): Unit = {
    logger.info(s"Mean AP = ${ "%.4f".format(aps.sum / aps.length) }")
    output.append(("Mean AP", aps.sum / aps.length))
    logger.info("~~~~~~~~")
    logger.info("Results:")
    aps.foreach(ap => logger.info(s"${ "%.3f".format(ap) }"))
    logger.info(s"${ "%.3f".format(aps.sum / aps.length) }")
    logger.info("~~~~~~~~")
  }
}

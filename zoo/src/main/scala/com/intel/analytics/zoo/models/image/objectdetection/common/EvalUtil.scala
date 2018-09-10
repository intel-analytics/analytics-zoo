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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel

import scala.collection.mutable.ArrayBuffer

object EvalUtil {

  /**
   * cumulate sum
   * @param arr array of values
   * @return cumulated sums
   */
  private def cumsum(arr: Array[Int]): Array[Int] = {
    var sum = 0
    arr.map { x => sum += x; sum }
  }

  /**
   * Compute VOC AP given precision and recall
   * @param rec recall
   * @param prec precision
   * @param use07metric whether use 07 11 point method
   * @return average precision
   */
  private def vocAp(rec: Array[Float], prec: Array[Float], use07metric: Boolean): Float = {
    var ap = 0.0f
    val num = rec.length
    if (use07metric) {
      val maxPrecs = new Array[Float](11)
      var startIdx = num - 1
      var j = 10
      while (j >= 0) {
        var i = startIdx
        var found = false
        while (i >= 0 && !found) {
          if (rec(i) < j / 10.0) {
            startIdx = i
            if (j > 0) {
              maxPrecs(j - 1) = maxPrecs(j)
            }
            found = true
          } else {
            if (maxPrecs(j) < prec(i)) {
              maxPrecs(j) = prec(i)
            }
          }
          i -= 1
        }
        j -= 1
      }
      j = 10
      while (j >= 0) {
        ap += maxPrecs(j) / 11
        j -= 1
      }
    } else {
      // correct AP calculation
      // first append sentinel values at the end
      val mrec = new Array[Float](rec.length + 2)
      mrec(mrec.length - 1) = 1.0f
      rec.copyToArray(mrec, 1)
      val mpre = new Array[Float](prec.length + 2)
      prec.copyToArray(mpre, 1)

      // compute the precision envelope
      var i = mpre.length - 1
      while (i > 0) {
        mpre(i - 1) = Math.max(mpre(i - 1), mpre(i))
        i -= 1
      }
      // to calculate area under PR curve, look for points
      // where X axis (recall) changes value
      val inds = (mrec.slice(1, mrec.length) zip mrec.slice(0, mrec.length - 1)).map(
        x => x._1 != x._2).zipWithIndex.map(x => x._2)

      // and sum (\Delta recall) * prec
      ap = inds.map(i => (mrec(i + 1) - mrec(i)) * mpre(i + 1)).sum
    }
    ap
  }

  def evaluateBatch(results: Array[Array[RoiLabel]], gt: Tensor[Float], gtAreas: Tensor[Float],
    clsInd: Int, ovThresh: Double = 0.5,
    use07Metric: Boolean = false, normalized: Boolean = false): (Int, Array[(Float, Int, Int)]) = {
    if (gt.nElement() > 0) require(gt.size(2) == 7)
    // extract gt objects for this class
    val num = results.length
    val labelGts = new Array[(Array[(Int, Float)], Array[Boolean])](num)

    val imgToDetectInds = new ArrayBuffer[(Int, Int)]()

    var npos = 0
    var i = 1
    val labelGtInds = new Array[ArrayBuffer[(Int, Float)]](num)
    var set = Set[Int]()
    var id = -1
    // var imgId = -1
    // assume the image ids are labeled from 0 for each batch
    // (imgId, label, diff, x1, y1, x2, y2)
    while (gt.nElement() > 0 && i <= gt.size(1)) {
      val imgId = gt.valueAt(i, 1).toInt
      if (!set.contains(imgId)) {
        set += imgId
        id += 1
      }
      if (gt.valueAt(i, 2) == clsInd + 1) {
        if (labelGtInds(id) == null) {
          labelGtInds(id) = new ArrayBuffer[(Int, Float)]()
        }
        if (gt.valueAt(i, 3) == 0) {
          npos += 1
        }
        labelGtInds(id).append((i, gt.valueAt(i, 3)))
      }
      i += 1
    }

    i = 0
    while (i < labelGtInds.length) {
      if (labelGtInds(i) != null) {
        val det = new Array[Boolean](labelGtInds(i).length)
        labelGts(i) = (labelGtInds(i).toArray, det)
      }
      i += 1
    }

    var imgInd = 0
    while (imgInd < num) {
      val output = results(imgInd)(clsInd)
      if (output != null && output.classes.nElement() != 0) {
        var i = 1
        while (i <= output.classes.size(1)) {
          imgToDetectInds.append((imgInd, i))
          i += 1
        }
      }
      imgInd += 1
    }

    val gtBoxes = if (gt.nElement() > 0) gt.narrow(2, 4, 4) else null
    val out = imgToDetectInds.map(box => {
      var tp = 0
      var fp = 0
      val labeledGt = labelGts(box._1)
      // No ground truth for current image. All detections become false_pos.
      val (ovmax, jmax) = if (gtBoxes == null || labeledGt == null) (-1f, -1)
      else {
        BboxUtil.getMaxOverlaps(gtBoxes, gtAreas, labeledGt._1.map(_._1),
          results(box._1)(clsInd).bboxes(box._2), normalized)
      }
      if (ovmax > ovThresh) {
        // not difficult
        if (labeledGt._1(jmax)._2 == 0) {
          if (!labeledGt._2(jmax)) {
            tp = 1
            // mark as visited
            labeledGt._2(jmax) = true
          } else {
            fp = 1
          }
        }
      } else {
        fp = 1
      }
      (results(box._1)(clsInd).classes.valueAt(box._2), tp, fp)
    }).toArray
    (npos, out)
  }

  /**
   * compute average precision
   * @param results (score, tp, fp)
   * @param use07Metric is true, use 11 points evaluation
   * @param nLabelPos number of positive lables
   * @return ap
   */
  def computeAP(results: Array[(Float, Int, Int)], use07Metric: Boolean, nLabelPos: Int)
  : Double = {
    val tp = results.map(x => (x._1, x._2)).sortBy(-_._1).map(_._2)
    val fp = results.map(x => (x._1, x._3)).sortBy(-_._1).map(_._2)

    // compute precision recall
    val cumfp = cumsum(fp)
    val cumtp = cumsum(tp)

    val rec = cumtp.map(x => x.toFloat / nLabelPos)
    val prec = (cumtp zip cumfp).map(x => {
      x._1.toFloat / (x._1 + x._2)
    })
    vocAp(rec, prec, use07Metric)
  }
}


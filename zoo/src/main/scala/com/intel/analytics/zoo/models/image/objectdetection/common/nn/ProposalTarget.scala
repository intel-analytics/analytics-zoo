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

package com.intel.analytics.zoo.models.image.objectdetection.common.nn

import breeze.numerics.round
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.models.image.objectdetection.common.BboxUtil
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.FrcnnMiniBatch
import com.intel.analytics.zoo.models.image.objectdetection.common.nn.ProposalTarget._
import org.apache.log4j.Logger

import scala.util.Random

object ProposalTarget {
  val logger = Logger.getLogger(getClass)

  // Normalize the targets using "precomputed" (or made up) means and stdevs
  // (BBOX_NORMALIZE_TARGETS must also be true)
  var BBOX_NORMALIZE_TARGETS_PRECOMPUTED = true

  val BBOX_NORMALIZE_MEANS = Tensor[Float](T(0.0f, 0.0f, 0.0f, 0.0f))
  val BBOX_NORMALIZE_STDS = Tensor[Float](T(0.1f, 0.1f, 0.2f, 0.2f))

  // Overlap threshold for a ROI to be considered background (class = 0 if
  // overlap in [LO, HI))
  val BG_THRESH_HI = 0.5
  val BG_THRESH_LO = 0.0

  def apply(batchSize: Int, numClasses: Int)
    (implicit ev: TensorNumeric[Float]): ProposalTarget =
    new ProposalTarget(batchSize, numClasses)
}

/**
 * Assign object detection proposals to ground-truth targets. Produces proposal
 * classification labels and bounding-box regression targets.
 * @param roiPerImage Minibatch size (number of regions of interest [ROIs])
 */
@SerialVersionUID(6250296121335715244L)
class ProposalTarget(val roiPerImage: Int, val numClasses: Int)
  (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Table, Float] {

  /**
   * Compute bounding-box regression targets for an image.
   *
   */
  private[objectdetection] def computeTargets(sampledRois: Tensor[Float],
    gtRois: Tensor[Float],
    labels: Tensor[Float]): Tensor[Float] = {

    val targets = BboxUtil.bboxTransform(sampledRois, gtRois)

    if (BBOX_NORMALIZE_TARGETS_PRECOMPUTED) {
      // Optionally normalize targets by a precomputed mean and stdev
      for (r <- 1 to targets.size(1)) {
        targets(r).add(-1, BBOX_NORMALIZE_MEANS)
        targets(r).cdiv(BBOX_NORMALIZE_STDS)
      }
    }
    BboxUtil.horzcat(labels.resize(labels.nElement(), 1), targets)
  }


  private var debug = false
  def setDebug(isDebug: Boolean): this.type = {
    debug = isDebug
    this
  }

  // Fraction of minibatch that is labeled foreground (i.e. class > 0)
  private val FG_FRACTION = 0.25
  private val fgRoisPerImage = round(FG_FRACTION * roiPerImage).toInt

  private var fgRoisPerThisImage = 0
  private var bgRoisPerThisImage = 0

  // Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
  private val FG_THRESH = 0.5f

  private def selectForeGroundRois(maxOverlaps: Tensor[Float]): Array[Int] = {
    // Select foreground RoIs as those with >= FG_THRESH overlap
    var fgInds = Array.range(1, maxOverlaps.nElement() + 1)
      .filter(i => maxOverlaps.valueAt(i, 1) >= FG_THRESH)
    // Guard against the case when an image has fewer than fg_rois_per_image
    // foreground RoIs
    fgRoisPerThisImage = Math.min(fgRoisPerImage, fgInds.length)
    // Sample foreground regions without replacement
    if (fgInds.length > 0) {
      fgInds = if (debug) {
        fgInds.toList.slice(0, fgRoisPerThisImage).toArray
      } else {
        Random.shuffle(fgInds.toList).slice(0, fgRoisPerThisImage).toArray
      }
    }
    fgInds
  }

  private def selectBackgroundRois(maxOverlaps: Tensor[Float]): Array[Int] = {
    // Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    var bgInds = Array.range(1, maxOverlaps.nElement() + 1)
      .filter(i => (maxOverlaps.valueAt(i, 1) < BG_THRESH_HI) &&
        (maxOverlaps.valueAt(i, 1) >= BG_THRESH_LO))
    // Compute number of background RoIs to take from this image (guarding
    // against there being fewer than desired)
    bgRoisPerThisImage = Math.min(roiPerImage - fgRoisPerThisImage, bgInds.length)
    // Sample background regions without replacement
    if (bgInds.length > 0) {
      bgInds = if (debug) {
        bgInds.toList.slice(0, bgRoisPerThisImage).toArray
      } else {
        Random.shuffle(bgInds.toList).slice(0, bgRoisPerThisImage).toArray
      }
    }
    bgInds
  }


  /**
   * Generate a random sample of RoIs comprising foreground and background examples.
   *
   * @param roisPlusGts (0, x1, y1, x2, y2)
   * @param gts GT boxes (index, label, difficult, x1, y1, x2, y2)
   * @return
   */
  private def sampleRois(roisPlusGts: Tensor[Float],
    gts: Tensor[Float])
  : (Tensor[Float], Tensor[Float], Tensor[Float], Tensor[Float]) = {
    // overlaps: (rois x gt_boxes)
    val overlaps = BboxUtil.bboxOverlap(roisPlusGts.narrow(2, 2, 4), FrcnnMiniBatch.getBboxes(gts))

    // for each roi, get the gt with max overlap with it
    val (maxOverlaps, gtIndices) = overlaps.max(2)
    // todo: the last few overlap should be 1, they are gt overlap gt

    // labels for rois
    var labels = Tensor[Float](gtIndices.nElement())
    (1 to gtIndices.nElement()).foreach(i => {
      val cls = gts.valueAt(gtIndices.valueAt(i, 1).toInt, FrcnnMiniBatch.labelIndex)
      require(cls >= 1 && cls <= numClasses, s"$cls is not in range [1, $numClasses]")
      labels.setValue(i, cls)
    })



    // from max overlaps, select foreground and background
    val fgInds = selectForeGroundRois(maxOverlaps)
    val bg_inds = selectBackgroundRois(maxOverlaps)
    // The indices that we're selecting (both fg and bg)
    val keepInds = fgInds ++ bg_inds

    // Select sampled values from various arrays:
    labels = BboxUtil.selectMatrix(labels, keepInds, 1)
    // Clamp labels for the background RoIs to 1 (1-based)
    (fgRoisPerThisImage + 1 to labels.nElement()).foreach(i => labels(i) = 1)

    val sampledRois = BboxUtil.selectMatrix(roisPlusGts, keepInds, 1)
    val keepInds2 = keepInds.map(x => gtIndices.valueAt(x, 1).toInt)

    val bboxTargetData = computeTargets(
      sampledRois.narrow(2, 2, 4),
      BboxUtil.selectMatrix(gts, keepInds2, 1)
        .narrow(2, FrcnnMiniBatch.x1Index, 4),
      labels)

    val (bboxTarget, bboxInsideWeights) =
      BboxUtil.getBboxRegressionLabels(bboxTargetData, numClasses)
    (labels.squeeze(), sampledRois, bboxTarget, bboxInsideWeights)
  }

  override def updateOutput(input: Table): Table = {
    if (!isTraining()) {
      output(1) = input(1)
      output(2) = input(2)
      return output
    }

    // Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    val proposalRois = input[Tensor[Float]](1)
    // GT boxes (index, label, difficult, x1, y1, x2, y2)
    val gts = input[Tensor[Float]](2)

    // Include ground-truth boxes in the set of candidate rois
    val roisPlusGts = BboxUtil.vertcat2D(proposalRois, gts.narrow(2, 3, 5))
    // in case gts has difficult (1)
    roisPlusGts.select(2, 1).fill(0)

    // Sample rois with classification labels and bounding box regression
    // targets
    val (labels, rois, bbox_targets, bboxInsideWeights) = sampleRois(roisPlusGts, gts)

    if (output.length() == 0) {
      bboxInsideWeights.apply1(x => {
        if (x > 0) 1f else 0f
      })
    }

    // sampled rois (0, x1, y1, x2, y2) (1,5)
    output.update(1, rois)
    // labels (1,1)
    output.update(2, labels)
    output.update(3, bbox_targets)
    output.update(4, bboxInsideWeights)
    output.update(5, bboxInsideWeights)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = null
    gradInput
  }
}

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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import AnchorTarget._
import com.intel.analytics.bigdl.nn.Anchor
import com.intel.analytics.zoo.models.image.objectdetection.common.BboxUtil


object AnchorTarget {
  val logger = Logger.getLogger(getClass)


  // Max number of foreground examples
  val RPN_FG_FRACTION = 0.5
  // IOU >= thresh: positive example
  val RPN_POSITIVE_OVERLAP = 0.7
  // IOU < thresh: negative example
  val RPN_NEGATIVE_OVERLAP = 0.3
  // If an anchor statisfied by positive and negative conditions set to negative
  private val RPN_CLOBBER_POSITIVES = false
  // Total number of examples
  val RPN_BATCHSIZE = 256
  private var debug = false
  // Deprecated (outside weights)
  val RPN_BBOX_INSIDE_WEIGHTS = Array(1.0f, 1.0f, 1.0f, 1.0f)
  // Give the positive RPN examples weight of p * 1 / {num positives}
  // and give negatives a weight of (1 - p)
  // Set to -1.0 to use uniform example weighting
  val RPN_POSITIVE_WEIGHT = -1.0f

  def apply(ratios: Array[Float], scales: Array[Float])
    (implicit ev: TensorNumeric[Float]): AnchorTarget = new AnchorTarget(ratios, scales)
}

@SerialVersionUID(-6245143347544093145L)
class AnchorTarget(val ratios: Array[Float], val scales: Array[Float])
  (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Table, Float] {
  @transient private var anchorTool: Anchor = _

  /**
   * Compute bounding-box regression targets for an image.
   *
   * @return
   */
  private def computeTargets(exRois: Tensor[Float], gtBoxes: Tensor[Float],
    insideAnchorsGtOverlaps: Tensor[Float]): Tensor[Float] = {
    val gtRois = BboxUtil.selectTensor(gtBoxes,
      BboxUtil.argmax2(insideAnchorsGtOverlaps, 2), 1)
    require(exRois.size(1) == gtRois.size(1))
    require(exRois.size(2) == 4)
    BboxUtil.bboxTransform(exRois, gtRois)
  }


  private def getAnchors(featureW: Int, featureH: Int, imgW: Int, imgH: Int)
  : (Array[Int], Tensor[Float], Int) = {
    if (anchorTool == null) {
      anchorTool = new Anchor(scales, ratios)
    }
    // 1. Generate proposals from bbox deltas and shifted anchors
    val anchorNum = ratios.length * scales.length
    val totalAnchors = featureH * featureW * anchorNum
    val allAnchors = anchorTool.generateAnchors(featureW, featureH)
    // keep only inside anchors
    val indsInside = getIndsInside(imgW, imgH, allAnchors, 0)
    val insideAnchors = Tensor[Float](indsInside.length, 4)
    indsInside.zip(Stream.from(1)).foreach(i => {
      insideAnchors.update(i._2, allAnchors(i._1))
    })
    (indsInside, insideAnchors, totalAnchors)
  }

  /**
   * Algorithm:
   *
   * for each (H, W) location i
   * --- generate scale_size * ratio_size anchor boxes centered on cell i
   * --- apply predicted bbox deltas at cell i to each of the 9 anchors
   * filter out-of-image anchors
   * measure GT overlap
   */
  private def getAnchorTarget(featureH: Int, featureW: Int,
    imgH: Int, imgW: Int, gtBoxes: Tensor[Float], output: Table): Unit = {
    require(gtBoxes.size(2) == 4, "gtBoxes is not a Nx4 tensor")
    if (anchorTool == null) {
      anchorTool = new Anchor(ratios, scales)
    }
    val anchorNum = ratios.length * scales.length
    val (indsInside, insideAnchors, totalAnchors) = getAnchors(featureW, featureH, imgW, imgH)

    // overlaps between the anchors and the gt boxes
    val insideAnchorsGtOverlaps = BboxUtil.bboxOverlap(insideAnchors, gtBoxes)

    // label: 1 is positive, 0 is negative, -1 is don't care
    var labels = getAllLabels(indsInside, insideAnchorsGtOverlaps)
    labels = sampleLabels(labels)
    // println(Tensor.sparse(labels + 1))
    var bboxTargets = computeTargets(insideAnchors, gtBoxes, insideAnchorsGtOverlaps)


    var bboxInsideWeights = getBboxInsideWeights(indsInside, labels)
    var bboxOutSideWeights = getBboxOutsideWeights(indsInside, labels)

    // map up to original set of anchors
    // mapUpToOriginal(labels, bboxTargets, bboxInsideWeights, bboxOutSideWeights, indsInside)
    labels = unmap(labels, totalAnchors, indsInside, -1)
    bboxTargets = unmap(bboxTargets, totalAnchors, indsInside, 0)
    bboxInsideWeights = unmap(bboxInsideWeights, totalAnchors, indsInside, 0)
    bboxOutSideWeights = unmap(bboxOutSideWeights, totalAnchors, indsInside, 0)

    labels = labels.reshape(Array(1, featureH, featureW, anchorNum))
      .transpose(2, 3).transpose(2, 4).reshape(Array(1, 1, anchorNum * featureH, featureW))
      .apply1(x => if (x == -1) -1 else x + 1f)

    bboxTargets = bboxTargets.reshape(Array(1, featureH, featureW, anchorNum * 4))
      .transpose(2, 3).transpose(2, 4)
    bboxInsideWeights = bboxInsideWeights.reshape(Array(1, featureH, featureW, anchorNum * 4))
      .transpose(2, 3).transpose(2, 4)
    bboxOutSideWeights = bboxOutSideWeights.reshape(Array(1, featureH,
      featureW, anchorNum * 4)).transpose(2, 3).transpose(2, 4)
    output.update(1, labels.contiguous())
    output.update(2, bboxTargets.contiguous())
    output.update(3, bboxInsideWeights.contiguous())
    output.update(4, bboxOutSideWeights.contiguous())
  }

  // label: 1 is positive, 0 is negative, -1 is don't care
  private def getAllLabels(indsInside: Array[Int],
    insideAnchorsGtOverlaps: Tensor[Float]): Tensor[Float] = {
    val labels = Tensor[Float](indsInside.length).fill(-1f)
    val argmaxOverlaps = BboxUtil.argmax2(insideAnchorsGtOverlaps, 2)
    val maxOverlaps = argmaxOverlaps.zip(Stream.from(1)).map(x =>
      insideAnchorsGtOverlaps.valueAt(x._2, x._1))
    val gtArgmaxOverlaps = BboxUtil.argmax2(insideAnchorsGtOverlaps, 1)

    val gtMaxOverlaps = gtArgmaxOverlaps.zip(Stream.from(1)).map(x => {
      insideAnchorsGtOverlaps.valueAt(x._1, x._2)
    })

    val gtArgmaxOverlaps2 = (1 to insideAnchorsGtOverlaps.size(1)).filter(r => {
      def isFilter: Boolean = {
        for (i <- 1 to insideAnchorsGtOverlaps.size(2)) {
          if (insideAnchorsGtOverlaps.valueAt(r, i) == gtMaxOverlaps(i - 1)) {
            return true
          }
        }
        false
      }
      isFilter
    })

    if (!RPN_CLOBBER_POSITIVES) {
      // assign bg labels first so that positive labels can clobber them
      maxOverlaps.zip(Stream from 1).foreach(x => {
        if (x._1 < RPN_NEGATIVE_OVERLAP) labels.setValue(x._2, 0)
      })
    }

    // fg label: for each gt, anchor with highest overlap
    gtArgmaxOverlaps2.foreach(x => labels.setValue(x, 1))

    // fg label: above threshold IOU
    maxOverlaps.zipWithIndex.foreach(x => {
      if (x._1 >= RPN_POSITIVE_OVERLAP) maxOverlaps(x._2) = 1
    })


    if (RPN_CLOBBER_POSITIVES) {
      // assign bg labels last so that negative labels can clobber positives
      maxOverlaps.zipWithIndex.foreach(x => {
        if (x._1 < RPN_NEGATIVE_OVERLAP) maxOverlaps(x._2) = 0
      })
    }
    labels
  }
  def setDebug(isDebug: Boolean): this.type = {
    debug = isDebug
    this
  }

  private def sampleLabels(labels: Tensor[Float]): Tensor[Float] = {
    // subsample positive labels if we have too many
    val numFg = RPN_FG_FRACTION * RPN_BATCHSIZE
    val fgInds = (1 to labels.size(1)).filter(x => labels.valueAt(x) == 1)
    if (fgInds.length > numFg) {
      val disableInds = if (!debug) {
        Random.shuffle(fgInds).take(fgInds.length - numFg.toInt)
      } else {
        fgInds.toList.slice(0, fgInds.length - numFg.toInt)
      }
      disableInds.foreach(x => labels.update(x, -1))
    }

    // subsample negative labels if we have too many
    val numBg = RPN_BATCHSIZE - fgInds.length
    val bgInds = (1 to labels.size(1)).filter(x => labels.valueAt(x) == 0)
    if (bgInds.length > numBg) {
      val disableInds = if (!debug) Random.shuffle(bgInds).take(bgInds.length - numBg.toInt)
      else bgInds.slice(0, bgInds.length - numBg.toInt)
      disableInds.foreach(x => labels.setValue(x, -1))
    }
    labels
  }

  private def getBboxInsideWeights(indsInside: Array[Int],
    labels: Tensor[Float]): Tensor[Float] = {
    val bboxInsideWeights = Tensor[Float](indsInside.length, 4)

    labels.map(Tensor[Float].range(1, labels.size(1)), (v, k) => {
      if (v == 1) {
        bboxInsideWeights.update(k.toInt, Tensor(Storage(RPN_BBOX_INSIDE_WEIGHTS)))
      }
      v
    })
    bboxInsideWeights
  }

  private def getBboxOutsideWeights(indsInside: Array[Int], labels: Tensor[Float])
  : Tensor[Float] = {
    val bboxOutSideWeights = Tensor[Float](indsInside.length, 4)

    val labelGe0 = (1 to labels.nElement()).filter(x => labels.valueAt(x) >= 0).toArray
    val labelE1 = (1 to labels.nElement()).filter(x => labels.valueAt(x) == 1).toArray
    val labelE0 = (1 to labels.nElement()).filter(x => labels.valueAt(x) == 0).toArray
    var positiveWeights: Tensor[Float] = null
    var negative_weights: Tensor[Float] = null
    if (RPN_POSITIVE_WEIGHT < 0) {
      // uniform weighting of examples (given non -uniform sampling)
      val numExamples = labelGe0.length
      positiveWeights = Tensor[Float](1, 4).fill(1f).mul(1.0f / numExamples)
      negative_weights = Tensor[Float](1, 4).fill(1f).mul(1.0f / numExamples)
    }
    else {
      require((RPN_POSITIVE_WEIGHT > 0) &
        (RPN_POSITIVE_WEIGHT < 1))
      positiveWeights = Tensor(Storage(Array(RPN_POSITIVE_WEIGHT / labelE1.length)))
      negative_weights = Tensor(Storage(Array(1.0f - RPN_POSITIVE_WEIGHT / labelE0.length)))
    }
    labelE1.foreach(x => bboxOutSideWeights.update(x, positiveWeights))
    labelE0.foreach(x => bboxOutSideWeights.update(x, negative_weights))
    bboxOutSideWeights
  }


  private def getIndsInside(width: Int, height: Int,
    allAnchors: Tensor[Float], allowedBorder: Float): Array[Int] = {
    var indsInside = ArrayBuffer[Int]()
    for (i <- 1 to allAnchors.size(1)) {
      if ((allAnchors.valueAt(i, 1) >= -allowedBorder) &&
        (allAnchors.valueAt(i, 2) >= -allowedBorder) &&
        (allAnchors.valueAt(i, 3) < width.toFloat + allowedBorder) &&
        (allAnchors.valueAt(i, 4) < height.toFloat + allowedBorder)) {
        indsInside += i
      }
    }
    indsInside.toArray
  }


  /**
   * Unmap a subset of item (data) back to the original set of items (of size count)
   */
  private def unmap(data: Tensor[Float], count: Int, inds: Array[Int],
    fillValue: Float): Tensor[Float] = {
    if (data.nDimension() == 1) {
      val ret = Tensor[Float](count).fill(fillValue)
      inds.zip(Stream.from(1)).foreach(ind => ret.setValue(ind._1, data.valueAt(ind._2)))
      ret
    } else {
      val ret = Tensor[Float](count, data.size(2)).fill(fillValue)
      inds.zip(Stream.from(1)).foreach(ind => {
        ret.update(ind._1, data(ind._2))
      })
      ret
    }
  }

  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   *
   * @param input
   * @return
   */
  override def updateOutput(input: Table): Table = {
    if (!isTraining()) return output
    val rpnReg = input[Tensor[Float]](1)
    val gtBoxes = input[Tensor[Float]](2).narrow(2, 4, 4)
    val imInfo = input[Tensor[Float]](3)
    getAnchorTarget(rpnReg.size(3), rpnReg.size(4),
      imInfo.valueAt(1, 1).toInt, imInfo.valueAt(1, 2).toInt, gtBoxes, output)
    output
  }

  /**
   * Computing the gradient of the module with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   *
   * @param input
   * @param gradOutput
   * @return
   */
  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = null
    gradOutput
  }
}

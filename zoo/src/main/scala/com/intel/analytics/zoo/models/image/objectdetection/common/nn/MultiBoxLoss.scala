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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, LogSoftMax, SmoothL1Criterion}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.models.image.objectdetection.common.BboxUtil
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

case class MultiBoxLossParam(locWeight: Double = 1.0,
  nClasses: Int = 21,
  shareLocation: Boolean = true,
  overlapThreshold: Double = 0.5,
  bgLabelInd: Int = 0,
  useDifficultGt: Boolean = true,
  negPosRatio: Double = 3.0,
  negOverlap: Double = 0.5)

class MultiBoxLoss[T: ClassTag](param: MultiBoxLossParam)
  (implicit ev: TensorNumeric[T]) extends AbstractCriterion[Table, Tensor[T], T] {
  @transient private var allLocPreds: Array[Array[Tensor[Float]]] = _
  @transient private var allIndices: Array[Array[Array[Int]]] = _
  @transient private var allIndicesNum: Array[Array[Int]] = _
  private val locPreds: Tensor[Float] = Tensor[Float]()
  private val locGts: Tensor[Float] = Tensor[Float]()
  private val locLoss = SmoothL1Criterion[Float](sizeAverage = false)
  private val confPreds: Tensor[Float] = Tensor[Float]()
  private val confGts: Tensor[Float] = Tensor[Float]()
  private val logSoftMax = LogSoftMax[Float]()
  private val classNLLCriterion = ClassNLLCriterion[Float](sizeAverage = false)

  private val locGradInput: Tensor[Float] = Tensor[Float]()
  private val confGradInput: Tensor[Float] = Tensor[Float]()
  private val priorBoxGradInput: Tensor[Float] = Tensor[Float]()
  gradInput.insert(locGradInput)
  gradInput.insert(confGradInput)
  gradInput.insert(priorBoxGradInput)

  private var allMatchIndices: Array[Array[Int]] = _
  private var allNegIndices: Array[Array[Int]] = _

  private var numMatches: Int = 0
  private var numConf: Int = 0

  private def init(batch: Int, numLocClasses: Int, nPriors: Int): Unit = {
    var i = 0
    if (allLocPreds == null || allLocPreds.length < batch) {
      // the outer array is the batch, each img contains an array of results, grouped by class
      allLocPreds = new Array[Array[Tensor[Float]]](batch)
      allIndices = new Array[Array[Array[Int]]](batch)
      allIndicesNum = new Array[Array[Int]](batch)
      i = 0
      while (i < batch) {
        allLocPreds(i) = new Array[Tensor[Float]](numLocClasses)
        allIndices(i) = new Array[Array[Int]](param.nClasses)
        allIndicesNum(i) = new Array[Int](param.nClasses)
        var c = 0
        while (c < numLocClasses) {
          allLocPreds(i)(c) = Tensor[Float](nPriors, 4)
          c += 1
        }
        c = 0
        while (c < param.nClasses) {
          if (c != param.bgLabelInd) allIndices(i)(c) = new Array[Int](nPriors)
          c += 1
        }
        i += 1
      }

    } else {
      i = 0
      while (i < batch) {
        var c = 0
        while (c < numLocClasses) {
          allLocPreds(i)(c).resize(nPriors, 4)
          c += 1
        }
        c = 0
        while (c < param.nClasses) {
          if (c != param.bgLabelInd && allIndices(i)(c).length < nPriors) {
            allIndices(i)(c) = new Array[Int](nPriors)
          }
          c += 1
        }
        i += 1
      }
    }
  }

  private var gtAreas: Tensor[Float] = _

  private def bipartiteMatching(gtInds: Array[Int], numPred: Int, allOverlaps: Array[Tensor[Float]],
    matchIndices: Array[Int], matchOverlaps: Array[Float]): Unit = {
    // Bipartite matching.
    var gtCount = gtInds.length
    while (gtCount > 0) {
      // Find the most overlapped gt and cooresponding predictions.
      var maxIdx = -1
      var maxGtIdx = -1
      var maxOverlap = -1f
      var i = 0
      while (i < numPred) {
        if (matchIndices(i) == -1) {
          var p = 0
          while (p < gtInds.length) {
            val j = gtInds(p)
            if (j != -1) {
              val overlap = allOverlaps(i).valueAt(j)
              if (overlap > 0) {
                // Find the maximum overlapped pair.
                if (overlap > maxOverlap) {
                  maxIdx = i
                  maxGtIdx = j - 1
                  maxOverlap = overlap
                }
              }
            }
            p += 1
          }
        }
        // else The prediction already has matched ground truth or is ignored.
        i += 1
      }
      if (maxIdx != -1) {
        require(matchIndices(maxIdx) == -1)
        matchIndices(maxIdx) = gtInds(maxGtIdx) - 1
        matchOverlaps(maxIdx) = maxOverlap
        // Erase the ground truth.
        gtInds(maxGtIdx) = -1
        gtCount = gtCount - 1
      } else {
        // Cannot find good match.
        return
      }
    }
  }

  private[objectdetection] def matchBbox(gtBoxes: Tensor[Float], predBoxes: Tensor[Float])
  : (Array[Int], Array[Float]) = {
    val numGt = gtBoxes.size(1)
    if (numGt == 0) return (new Array[Int](0), new Array[Float](0))
    if (gtAreas == null) gtAreas = Tensor[Float]()
    gtAreas.resize(gtBoxes.size(1))
    BboxUtil.getAreas(gtBoxes, gtAreas, 4, true)
    val numPred = predBoxes.size(1)
    val matchIndices = new Array[Int](numPred)
    val matchOverlaps = new Array[Float](numPred)
    var i = 0
    while (i < numPred) {
      matchIndices(i) = -1
      i += 1
    }
    i = 0
    while (i < numPred) {
      matchOverlaps(i) = 0
      i += 1
    }
    val gtPool = new Array[Int](numGt)
    val gtIndices = new Array[Int](numGt)
    i = 0
    while (i < gtIndices.length) {
      gtIndices(i) = i + 1
      gtPool(i) = i + 1
      i += 1
    }

    // Store the positive overlap between predictions and ground truth.
    val allOverlaps = new Array[Tensor[Float]](numPred)
    i = 0
    while (i < numPred) {
      val overlaps = Tensor[Float](gtAreas.size(1))
      val out = BboxUtil.getMaxOverlaps(gtBoxes, gtAreas, gtIndices,
        predBoxes(i + 1), true, overlaps, 3)
      matchOverlaps(i) = out._1
      allOverlaps(i) = overlaps
      i += 1
    }
    bipartiteMatching(gtPool, numPred, allOverlaps, matchIndices, matchOverlaps)

    // Get most overlaped for the rest prediction bboxes.
    i = 0
    while (i < numPred) {
      if (matchIndices(i) == -1) {
        var maxGtIdx = -1
        var maxOverlap = -1f
        var j = 0
        while (j < numGt) {
          val overlap = allOverlaps(i).valueAt(j + 1)
          if (overlap > 0 && overlap >= param.overlapThreshold && overlap > maxOverlap) {
            // If the prediction has not been matched to any ground truth,
            // and the overlap is larger than maximum overlap, update.
            maxGtIdx = j
            maxOverlap = overlap
          }
          j += 1
        }
        if (maxGtIdx != -1) {
          // Found a matched ground truth.
          require(matchIndices(i) == -1)
          matchIndices(i) = gtIndices(maxGtIdx) - 1
          assert(gtIndices(maxGtIdx) - 1 < numGt)
          matchOverlaps(i) = maxOverlap
        }
      }
      i += 1
    }

    (matchIndices, matchOverlaps)
  }

  /**
   * Find matches between prediction bboxes and ground truth bboxes.
   * @param allLocPreds stores the location prediction, where each item contains
   * location prediction for an image.
   * @param target stores ground truth bboxes for the batch.
   * @param priorBoxes
   * @param priorVariances
   */
  private def findMatches(batch: Int, allLocPreds: Array[Array[Tensor[Float]]],
    target: Map[Int, Tensor[Float]],
    priorBoxes: Tensor[Float], priorVariances: Tensor[Float])
  : (Array[Array[Int]], Array[Array[Float]]) = {
    allMatchIndices = new Array[Array[Int]](batch)
    val allMatchOverlaps = new Array[Array[Float]](batch)
    var i = 0
    while (i < batch) {
      // Check if there is ground truth for current image.
      if (target.contains(i) && target(i).size(1) > 0) {
        // Use prior bboxes to match against all ground truth.
        val gtBoxes = target(i)
        val (matchIndices, matchOverlaps) = matchBbox(gtBoxes, priorBoxes)

        allMatchIndices(i) = matchIndices
        allMatchOverlaps(i) = matchOverlaps
      }
      i += 1
    }
    (allMatchIndices, allMatchOverlaps)
  }

  private def countNumMatches(allMatchIndices: Array[Array[Int]]): Int = {
    var count = 0
    var i = 0
    while (i < allMatchIndices.length) {
      val matchIndices = allMatchIndices(i)
      if (matchIndices != null) {
        var j = 0
        while (j < matchIndices.length) {
          if (matchIndices(j) != -1) {
            count += 1
          }
          j += 1
        }
      }
      i += 1
    }
    count
  }


  private[nn] def computeConfLoss(conf: Tensor[Float], num: Int, numPredsPerClass: Int,
    nClasses: Int, bgIndex: Int, allMatchIndices: Array[Array[Int]],
    allGroundTruth: Map[Int, Tensor[Float]]): Array[Array[Float]] = {
    val allConfLoss = new Array[Array[Float]](num)
    var i = 0
    val confData = conf.storage().array()
    var confOffset = conf.storageOffset() - 1
    while (i < num) {
      val confLoss = new Array[Float](numPredsPerClass)
      val matchIndices = allMatchIndices(i)
      var p = 0
      while (p < numPredsPerClass) {
        val startIdx = confOffset + p * nClasses
        var labelInd = bgIndex
        if (matchIndices != null && matchIndices(p) > -1) {
          require(allGroundTruth.contains(i))
          labelInd = allGroundTruth(i).valueAt(matchIndices(p) + 1, 2).toInt - 1
          require(labelInd >= 0 && labelInd != bgIndex && labelInd < nClasses)
        }
        var loss = 0f
        // loss type softmax
        // Compute softmax probability.
        // We need to subtract the max to avoid numerical issues.
        var maxVal = confData(startIdx)
        var c = 1
        while (c < nClasses) {
          maxVal = Math.max(confData(startIdx + c), maxVal)
          c += 1
        }
        var sum = 0f
        c = 0
        while (c < nClasses) {
          sum += Math.exp(confData(startIdx + c) - maxVal).toFloat
          c += 1
        }
        val prob = Math.exp(confData(startIdx + labelInd) - maxVal) / sum
        loss = -Math.log(Math.max(prob, Float.MinPositiveValue)).toFloat
        confLoss(p) = loss
        p += 1
      }
      confOffset += numPredsPerClass * nClasses
      allConfLoss(i) = confLoss
      i += 1
    }
    allConfLoss
  }

  private def isEligibleMining(matchIdx: Int, matchOverlap: Float, negOverlap: Double): Boolean = {
    matchIdx == -1 && matchOverlap < negOverlap
  }

  private def mineHardExamples(conf: Tensor[Float], allLocPreds: Array[Array[Tensor[Float]]],
    allGroundTruth: Map[Int, Tensor[Float]], priorBoxes: Tensor[Float],
    priorVariances: Tensor[Float], allMatchOverlaps: Array[Array[Float]],
    allMatchIndices: Array[Array[Int]]): (Int, Array[Array[Int]]) = {
    val batch = conf.size(1)
    allNegIndices = new Array[Array[Int]](batch)
    var numNeg = 0
    val numPriors = priorBoxes.size(1)
    require(numPriors == priorVariances.size(1))
    val allConfLoss = computeConfLoss(conf, batch, numPriors, param.nClasses, param.bgLabelInd,
      allMatchIndices, allGroundTruth)

    var i = 0
    while (i < batch) {
      // Pick negatives or hard examples based on loss.
      val selectIndices = new ArrayBuffer[Int]()
      val negativeIndices = new ArrayBuffer[Int]()
      val matchedIndices = allMatchIndices(i)
      val matchedOverlaps = allMatchOverlaps(i)
      var numSel = 0
      // No localization loss.
      val loss = allConfLoss(i)
      // Get potential indices and loss pairs.
      val lossIndices = new ArrayBuffer[(Float, Int)]()
      var numPos = 0
      var m = 0
      if (matchedIndices != null) {
        while (m < matchedIndices.length) {
          if (isEligibleMining(matchedIndices(m), matchedOverlaps(m), param.negOverlap)) {
            lossIndices.append((loss(m), m))
            numSel += 1
          }
          if (matchedIndices(m) > -1) numPos += 1
          m += 1
        }
      }
      numSel = Math.min((numPos * param.negPosRatio).toInt, numSel)

      // Select samples.
      // Pick top example indices based on loss.
      val sortedLossIndices = lossIndices.sortBy(-_._1)
      var n = 0
      while (n < numSel) {
        selectIndices.append(sortedLossIndices(n)._2)
        n += 1
      }

      // Update the match_indices and select neg_indices.
      if (matchedIndices != null) {
        m = 0
        while (m < matchedIndices.length) {
          if (matchedIndices(m) == -1) {
            if (selectIndices.contains(m)) {
              negativeIndices.append(m)
              numNeg += 1
            }
          }
          m += 1
        }
      }
      allNegIndices(i) = negativeIndices.toArray
      i += 1
    }
    (numNeg, allNegIndices)
  }

  private def encodeLocPrediction(batch: Int, allLocPreds: Array[Array[Tensor[Float]]],
    allGroundTruth: Map[Int, Tensor[Float]], allMatchIndices: Array[Array[Int]],
    priorBoxes: Tensor[Float], priorVariances: Tensor[Float], locPreds: Tensor[Float],
    locGts: Tensor[Float]): Unit = {
    var count = 0
    var i = 0
    while (i < batch) {
      val matchIndices = allMatchIndices(i)
      val locPred = allLocPreds(i)
      var j = 0
      while (matchIndices != null && j < matchIndices.length) {
        if (matchIndices(j) > -1) {
          // Store encoded ground truth.
          val gtIdx = matchIndices(j)
          require(allGroundTruth.contains(i), s"the ${ i }th" +
            s" image must have gt")
          require(gtIdx >= 0 && gtIdx < allGroundTruth(i).size(1),
            s"$i $j gt index $gtIdx should less than " +
            s"gt length ${ allGroundTruth(i).size(1) }")
          val gtBox = allGroundTruth(i)(gtIdx + 1)
          BboxUtil.encodeBBox(priorBoxes(j + 1), priorVariances(j + 1), gtBox, locGts(count + 1))
          locPreds.setValue(count + 1, 1, locPred(0).valueAt(j + 1, 1))
          locPreds.setValue(count + 1, 2, locPred(0).valueAt(j + 1, 2))
          locPreds.setValue(count + 1, 3, locPred(0).valueAt(j + 1, 3))
          locPreds.setValue(count + 1, 4, locPred(0).valueAt(j + 1, 4))
          count += 1
        }
        j += 1
      }
      i += 1
    }
  }

  private def encodeConfPrediction(conf: Tensor[Float], batch: Int, nPriors: Int,
    allMatchIndices: Array[Array[Int]], allNegIndices: Array[Array[Int]],
    allGroundTruth: Map[Int, Tensor[Float]],
    confPreds: Tensor[Float], confGts: Tensor[Float]): Unit = {
    var i = 0
    var count = 0
    val confData = conf.storage().array()
    var confStart = conf.storageOffset() - 1
    val confPredsData = confPreds.storage().array()
    while (i < batch) {
      if (allGroundTruth.contains(i)) {
        // Save matched (positive) bboxes scores and labels.
        val matchIndices = allMatchIndices(i)
        require(matchIndices.length == nPriors)
        var j = 0
        while (j < nPriors) {
          if (matchIndices(j) > -1) {
            val gtLabel = allGroundTruth(i).valueAt(matchIndices(j) + 1, 2)
            confGts.setValue(count + 1, gtLabel)
            // Copy scores for matched bboxes.
            System.arraycopy(confData, confStart + j * param.nClasses,
              confPredsData, confPreds.storageOffset() - 1 + count * param.nClasses, param.nClasses)
            count += 1
          }
          j += 1
        }
        // Go to next image.
        // Save negative bboxes scores and labels.
        var n = 0
        while (n < allNegIndices(i).length) {
          val j = allNegIndices(i)(n)
          require(j < nPriors)
          System.arraycopy(confData, confStart + j * param.nClasses,
            confPredsData, confPreds.storageOffset() - 1 + count * param.nClasses, param.nClasses)
          confGts.setValue(count + 1, param.bgLabelInd + 1)
          n += 1
          count += 1
        }
      }
      confStart += nPriors * param.nClasses
      i += 1
    }
  }

  override def updateOutput(input: Table, target: Tensor[T]): T = {
    val loc = input[Tensor[Float]](1)
    val conf = input[Tensor[Float]](2)
    val prior = input[Tensor[Float]](3)
    val batch = loc.size(1)
    val numLocClasses = if (param.shareLocation) 1 else param.nClasses
    val nPriors = prior.size(3) / 4

    init(batch, numLocClasses, nPriors)
    // Retrieve all predictions.
    BboxUtil.getLocPredictions(loc, nPriors, numLocClasses, param.shareLocation,
      allLocPreds)

    val (priorBoxes, priorVariances) = BboxUtil.getPriorBboxes(prior, nPriors)

    val allGroundTruth = BboxUtil.getGroundTruths(target.asInstanceOf[Tensor[Float]])
    // Find matches between source bboxes and ground truth bboxes.
    val (allMatchIndices, allMatchOverlaps) =
    findMatches(batch, allLocPreds, allGroundTruth, priorBoxes, priorVariances)


    numMatches = countNumMatches(allMatchIndices)

    // Sample hard negative (and positive) examples based on mining type.
    val (numNeg, allNegIndices) =
    mineHardExamples(conf, allLocPreds, allGroundTruth, priorBoxes,
      priorVariances, allMatchOverlaps, allMatchIndices)

    val locLossOut = if (numMatches >= 1) {
      // Form data to pass on to loc_loss_layer_.
      locPreds.resize(numMatches, 4)
      locGts.resize(numMatches, 4)
      encodeLocPrediction(batch, allLocPreds, allGroundTruth, allMatchIndices, priorBoxes,
        priorVariances, locPreds, locGts)
      locLoss.forward(locPreds, locGts)
    } else {
      0
    }
    // Form data to pass on to conf_loss_layer_.
    numConf = numMatches + numNeg
    val confLossOut = if (numConf >= 1) {
      confGts.resize(numConf).fill(param.bgLabelInd + 1)
      confPreds.resize(numConf, param.nClasses)
      encodeConfPrediction(conf, batch, nPriors, allMatchIndices, allNegIndices,
        allGroundTruth, confPreds, confGts)
      logSoftMax.forward(confPreds)
      classNLLCriterion.forward(logSoftMax.output, confGts)
    } else {
      0
    }
    output = if (numMatches == 0) {
      ev.fromType(0)
    } else {
      ev.fromType((param.locWeight * locLossOut + confLossOut) / numMatches)
    }
    output
  }

  override def updateGradInput(input: Table, target: Tensor[T]): Table = {
    // Back propagate on location prediction.
    val loc = input[Tensor[Float]](1)
    val conf = input[Tensor[Float]](2)
    val prior = input[Tensor[Float]](3)
    val batch = loc.size(1)
    locGradInput.resizeAs(loc).zero()
    confGradInput.resizeAs(conf).zero()
    if (priorBoxGradInput.nElement() != prior.nElement()) {
      priorBoxGradInput.resizeAs(prior).zero()
    }
    if (ev.isGreater(output, ev.fromType(50))) {
      return gradInput
    }

    if (numMatches >= 1) {
      locLoss.backward(locPreds, locGts)
      locLoss.gradInput.div(numMatches).mul(param.locWeight.toFloat)
      val locPredDiff = locLoss.gradInput.storage().array()
      val locPredStart = locLoss.gradInput.storageOffset() - 1


      val locGradData = locGradInput.storage().array()
      var locGradStart = locGradInput.storageOffset() - 1

      // Copy gradient back to locGradInput
      var i = 0
      var count = 0
      while (i < batch) {
        val matchIndices = allMatchIndices(i)
        var j = 0
        while (matchIndices != null && j < matchIndices.length) {
          if (matchIndices(j) > -1) {
            // Copy the diff to the right place.
            Array.copy(locPredDiff, locPredStart + count * 4,
              locGradData, locGradStart + 4 * j, 4)
            count += 1
          }
          j += 1
        }
        locGradStart += locGradInput.stride(1)
        i += 1
      }
    }
    if (numConf >= 1) {
      classNLLCriterion.backward(logSoftMax.output, confGts)
      logSoftMax.backward(confPreds, classNLLCriterion.gradInput)
      logSoftMax.gradInput.div(numMatches)
      val confPredDiff = logSoftMax.gradInput.storage().array()
      val confPredStart = logSoftMax.gradInput.storageOffset() - 1

      val confGradData = confGradInput.storage().array()
      var confGradStart = confGradInput.storageOffset() - 1
      var i = 0
      var count = 0
      // Copy gradient back to confGradInput
      while (i < batch) {
        val matchIndices = allMatchIndices(i)
        var j = 0
        while (matchIndices != null && j < matchIndices.length) {
          if (matchIndices(j) > -1) {
            // Copy the diff to the right place.
            Array.copy(confPredDiff, confPredStart + count * param.nClasses,
              confGradData, confGradStart + j * param.nClasses, param.nClasses)
            count += 1
          }
          j += 1
        }
        // Copy negative bboxes scores' diff.
        var n = 0
        while (n < allNegIndices(i).length) {
          val t = allNegIndices(i)(n)
          Array.copy(confPredDiff, confPredStart + count * param.nClasses,
            confGradData, confGradStart + t * param.nClasses, param.nClasses)
          count += 1
          n += 1
        }
        confGradStart += confGradInput.stride(1)
        i += 1
      }
    }
    gradInput
  }
}

object MultiBoxLoss {
  val logger = Logger.getLogger(getClass.getName)
  def apply[@specialized(Float, Double) T: ClassTag](param: MultiBoxLossParam)
    (implicit ev: TensorNumeric[T]): MultiBoxLoss[T] = new MultiBoxLoss[T](param)
}

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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.util.BoundingBox
import org.apache.log4j.Logger

import scala.reflect.ClassTag

object BboxUtil {

  def selectTensor(matrix: Tensor[Float], indices: Array[Int], dim: Int, indiceLen: Int = -1,
    out: Tensor[Float] = null): Tensor[Float] = {
    assert(dim == 1 || dim == 2)
    var i = 1
    val n = if (indiceLen == -1) indices.length else indiceLen
    if (matrix.nDimension() == 1) {
      val res = if (out == null) {
        Tensor[Float](n)
      } else {
        out.resize(n)
      }
      while (i <= n) {
        res.update(i, matrix.valueAt(indices(i - 1)))
        i += 1
      }
      return res
    }
    // select rows
    if (dim == 1) {
      val res = if (out == null) {
        Tensor[Float](n, matrix.size(2))
      } else {
        out.resize(n, matrix.size(2))
      }
      while (i <= n) {
        res.update(i, matrix(indices(i - 1)))
        i += 1
      }
      res
    } else {
      val res = if (out == null) {
        Tensor[Float](matrix.size(1), n)
      } else {
        out.resize(matrix.size(1), n)
      }
      while (i <= n) {
        var rid = 1
        val value = matrix.select(2, indices(i - 1))
        while (rid <= res.size(1)) {
          res.setValue(rid, i, value.valueAt(rid))
          rid += 1
        }
        i += 1
      }
      res
    }
  }

  /**
   * return the max value in rows(d=0) or in cols(d=1)
   * arr = [4 9
   * 5 7
   * 8 5]
   *
   * argmax2(arr, 1) will return 3, 1
   * argmax2(arr, 2) will return 2, 2, 1
   * @return
   * todo: this maybe removed
   */
  def argmax2(arr: Tensor[Float], d: Int): Array[Int] = {
    require(d >= 1)
    arr.max(d)._2.storage().array().map(x => x.toInt)
  }

  /**
   * Bounding-box regression targets (bboxTargetData) are stored in a
   * compact form N x (class, tx, ty, tw, th)
   * *
   * This function expands those targets into the 4-of-4*K representation used
   * by the network (i.e. only one class has non-zero targets).
   * *
   * Returns:
   * bbox_target (ndarray): N x 4K blob of regression targets
   * bbox_inside_weights (ndarray): N x 4K blob of loss weights
   *
   */
  def getBboxRegressionLabels(bboxTargetData: Tensor[Float],
    numClasses: Int): (Tensor[Float], Tensor[Float]) = {
    // Deprecated (inside weights)
    val BBOX_INSIDE_WEIGHTS = Tensor(Storage(Array(1.0f, 1.0f, 1.0f, 1.0f)))
    val bboxTargets = Tensor[Float](bboxTargetData.size(1), 4 * numClasses)
    val bboxInsideWeights = Tensor[Float]().resizeAs(bboxTargets)
    (1 to bboxTargetData.size(1)).foreach(ind => {
      val cls = bboxTargetData.valueAt(ind, 1)
      if (cls > 1) {
        require(cls <= numClasses, s"$cls is not in range [1, $numClasses]")
        val start = (4 * (cls - 1)).toInt

        (2 to bboxTargetData.size(2)).foreach(x => {
          bboxTargets.setValue(ind, x + start - 1, bboxTargetData.valueAt(ind, x))
          bboxInsideWeights.setValue(ind, x + start - 1,
            BBOX_INSIDE_WEIGHTS.valueAt(x - 1))
        })
      }
    })
    (bboxTargets, bboxInsideWeights)
  }

  def vertcat2D[T: ClassTag](tensors: Tensor[T]*)
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(tensors(0).dim() == 2, "only support 2D")
    var nRows = tensors(0).size(1)
    val nCols = tensors(0).size(2)
    for (i <- 1 until tensors.length) {
      require(tensors(i).size(2) == nCols, "the cols length must be equal")
      nRows += tensors(i).size(1)
    }
    val resData = Tensor[T](nRows, nCols)
    var id = 0
    tensors.foreach { tensor =>
      (1 to tensor.size(1)).foreach(rid => {
        id = id + 1
        resData.update(id, tensor(rid))
      })
    }
    resData
  }

  def vertcat1D[T: ClassTag](tensors: Tensor[T]*)
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(tensors(0).dim() == 1, "only support 1D")
    val nCols = tensors(0).size(1)
    for (i <- 1 until tensors.length) {
      require(tensors(i).size(1) == nCols, "the cols length must be equal")
    }
    val resData = Tensor[T](tensors.size, nCols)
    var id = 0
    tensors.foreach { tensor =>
      id = id + 1
      resData.update(id, tensor)
    }
    resData
  }


  def horzcat(tensors: Tensor[Float]*): Tensor[Float] = {
    require(tensors(0).dim() == 2, "currently only support 2D")
    val nRows = tensors(0).size(1)
    var nCols = tensors(0).size(2)
    for (i <- 1 until tensors.length) {
      require(tensors(i).size(1) == nRows, "the rows length must be equal")
      nCols += tensors(i).size(2)
    }
    val resData = Tensor[Float](nRows, nCols)
    var id = 1
    tensors.foreach { tensor =>
      updateRange(resData, 1, nRows, id, id + tensor.size(2) - 1, tensor)
      id = id + tensor.size(2)
    }
    resData
  }

  /**
   * update with 2d tensor, the range must be equal to the src tensor size
   *
   */
  def updateRange(dest: Tensor[Float], startR: Int, endR: Int, startC: Int, endC: Int,
    src: Tensor[Float]): Unit = {
    assert(src.size(1) == endR - startR + 1)
    assert(src.size(2) == endC - startC + 1)
    (startR to endR).zip(Stream.from(1)).foreach(r => {
      (startC to endC).zip(Stream.from(1)).foreach(c => {
        dest.setValue(r._1, c._1, src.valueAt(r._2, c._2))
      })
    })
  }


  /**
   *
   * @param boxes (N, 4) ndarray of float
   * @param queryBoxes (K, >=4) ndarray of float
   * @return overlaps: (N, K) ndarray of overlap between boxes and query_boxes
   */
  def bboxOverlap(boxes: Tensor[Float], queryBoxes: Tensor[Float]): Tensor[Float] = {
    require(boxes.size(2) >= 4)
    require(queryBoxes.size(2) >= 4)
    val N = boxes.size(1)
    val K = queryBoxes.size(1)
    val overlaps = Tensor[Float](N, K)

    for (k <- 1 to K) {
      val boxArea = (queryBoxes.valueAt(k, 3) - queryBoxes.valueAt(k, 1) + 1) *
        (queryBoxes.valueAt(k, 4) - queryBoxes.valueAt(k, 2) + 1)
      for (n <- 1 to N) {
        val iw = Math.min(boxes.valueAt(n, 3), queryBoxes.valueAt(k, 3)) -
          Math.max(boxes.valueAt(n, 1), queryBoxes.valueAt(k, 1)) + 1
        if (iw > 0) {
          val ih = Math.min(boxes.valueAt(n, 4), queryBoxes.valueAt(k, 4)) -
            Math.max(boxes.valueAt(n, 2), queryBoxes.valueAt(k, 2)) + 1

          if (ih > 0) {
            val ua = (boxes.valueAt(n, 3) - boxes.valueAt(n, 1) + 1) *
              (boxes.valueAt(n, 4) - boxes.valueAt(n, 2) + 1) + boxArea - iw * ih
            overlaps.setValue(n, k, iw * ih / ua)
          }
        }
      }
    }
    overlaps
  }

  def selectMatrix(matrix: Tensor[Float], indices: Array[Int], dim: Int, indiceLen: Int = -1,
    out: Tensor[Float] = null): Tensor[Float] = {
    assert(dim == 1 || dim == 2)
    var i = 1
    val n = if (indiceLen == -1) indices.length else indiceLen
    if (matrix.nDimension() == 1) {
      val res = if (out == null) {
        Tensor[Float](n)
      } else {
        out.resize(n)
      }
      while (i <= n) {
        res.update(i, matrix.valueAt(indices(i - 1)))
        i += 1
      }
      return res
    }
    // select rows
    if (dim == 1) {
      val res = if (out == null) {
        Tensor[Float](n, matrix.size(2))
      } else {
        out.resize(n, matrix.size(2))
      }
      while (i <= n) {
        res.update(i, matrix(indices(i - 1)))
        i += 1
      }
      res
    } else {
      val res = if (out == null) {
        Tensor[Float](matrix.size(1), n)
      } else {
        out.resize(matrix.size(1), n)
      }
      while (i <= n) {
        var rid = 1
        val value = matrix.select(2, indices(i - 1))
        while (rid <= res.size(1)) {
          res.setValue(rid, i, value.valueAt(rid))
          rid += 1
        }
        i += 1
      }
      res
    }
  }

  def selectMatrix2(mat: Tensor[Float],
    rows: Array[Int], cols: Array[Int]): Tensor[Float] = {
    val out = Tensor[Float](rows.length, cols.length)
    rows.zip(Stream from 1).map(r => {
      cols.zip(Stream from 1).map(c => {
        out.setValue(r._2, c._2, mat.valueAt(r._1, c._1))
      })
    })
    out
  }

  def bboxTransform(sampledRois: Tensor[Float], gtRois: Tensor[Float]): Tensor[Float] = {
    require(sampledRois.size(1) == gtRois.size(1), "each sampledRois should have a gtRoi")
    require(sampledRois.size(2) == 4)
    require(gtRois.size(2) == 4)
    val transformed = Tensor[Float].resizeAs(sampledRois).copy(sampledRois)

    (1 to sampledRois.size(1)).foreach(i => {
      val sampleWidth = sampledRois.valueAt(i, 3) - sampledRois.valueAt(i, 1) + 1.0f
      val sampleHeight = sampledRois.valueAt(i, 4) - sampledRois.valueAt(i, 2) + 1.0f
      val sampleCtrX = sampledRois.valueAt(i, 1) + sampleWidth * 0.5f
      val sampleCtrY = sampledRois.valueAt(i, 2) + sampleHeight * 0.5f

      val gtWidth = gtRois.valueAt(i, 3) - gtRois.valueAt(i, 1) + 1.0f
      val gtHeight = gtRois.valueAt(i, 4) - gtRois.valueAt(i, 2) + 1.0f
      val gtCtrX = gtRois.valueAt(i, 1) + gtWidth * 0.5f
      val gtCtrY = gtRois.valueAt(i, 2) + gtHeight * 0.5f

      // targetsDx
      transformed.setValue(i, 1, (gtCtrX - sampleCtrX) / sampleWidth)
      // targetsDy
      transformed.setValue(i, 2, (gtCtrY - sampleCtrY) / sampleHeight)
      // targetsDw
      transformed.setValue(i, 3, Math.log(gtWidth / sampleWidth).toFloat)
      // targetsDh
      transformed.setValue(i, 4, Math.log(gtHeight / sampleHeight).toFloat)
    })

    transformed
  }


  def resultToString(output: Tensor[Float], path: String, toInt: Boolean = true): String = {
    val decoded = decodeRois(output)
    if (decoded.nElement() == 0) ""
    else {
      (1 to decoded.size(1)).map(i => {
        if (toInt) {
          path +
            " " + decoded.valueAt(i, 1).toInt +
            " " + decoded.valueAt(i, 2) +
            " " + decoded.valueAt(i, 3).round +
            " " + decoded.valueAt(i, 4).round +
            " " + decoded.valueAt(i, 5).round +
            " " + decoded.valueAt(i, 6).round
        } else {
          path +
            " " + decoded.valueAt(i, 1).toInt +
            " " + decoded.valueAt(i, 2) +
            " " + decoded.valueAt(i, 3) +
            " " + decoded.valueAt(i, 4) +
            " " + decoded.valueAt(i, 5) +
            " " + decoded.valueAt(i, 6)
        }
      }).mkString("\n")
    }
  }

  // decode batch
  def decodeBatchOutput(output: Tensor[Float], nClass: Int): Array[Array[RoiLabel]] = {
    var i = 0
    val batch = output.size(1)
    val decoded = new Array[Array[RoiLabel]](batch)
    while (i < batch) {
      decoded(i) = decodeRois(output(i + 1), nClass)
      i += 1
    }
    decoded
  }

  def decodeRois(output: Tensor[Float], nclass: Int)
  : Array[RoiLabel] = {
    val result = decodeRois(output)
    val indices = getClassIndices(result)
    val decoded = new Array[RoiLabel](nclass)
    val iter = indices.iterator
    while (iter.hasNext) {
      val item = iter.next()
      val sub = result.narrow(1, item._2._1, item._2._2)
      decoded(item._1) = RoiLabel(sub.select(2, 2),
        sub.narrow(2, 3, 4))
    }
    decoded
  }

  def decodeRois(output: Tensor[Float]): Tensor[Float] = {
    val num = output.valueAt(1).toInt
    require(num >= 0, "output number should >= 0")
    if (num == 0) {
      Tensor[Float]()
    } else {
      output.narrow(1, 2, num * 6).view(num, 6)
    }
  }

  def scaleBatchOutput(output: Tensor[Float], imInfo: Tensor[Float])
  : Tensor[Float] = {
    val batch = output.size(1)
    require(batch == imInfo.size(1),
      "the batch size of output is not equal to the batch size of imInfo")
    var i = 0
    while (i < batch) {
      // Scale the bbox according to the original image size.
      val height = imInfo.valueAt(i + 1, 1) * imInfo.valueAt(i + 1, 3)
      val width = imInfo.valueAt(i + 1, 2) * imInfo.valueAt(i + 1, 4)
      val single = output(i + 1)
      val result = decodeRois(single)
      if (result.nElement() > 0) {
        // clipBoxes to [0, 1]
        clipBoxes(result.narrow(2, 3, 4))
        // scaleBoxes
        result.select(2, 3).mul(width)
        result.select(2, 4).mul(height)
        result.select(2, 5).mul(width)
        result.select(2, 6).mul(height)
      }
      i += 1
    }
    output
  }

  private def getClassIndices(result: Tensor[Float]): Map[Int, (Int, Int)] = {
    var indices = Map[Int, (Int, Int)]()
    if (result.nElement() == 0) return indices
    var prev = -1f
    var i = 1
    var start = 1
    if (result.size(1) == 1) {
      indices += (result.valueAt(i, 1).toInt -> (1, 1))
      return indices
    }
    while (i <= result.size(1)) {
      if (prev != result.valueAt(i, 1)) {
        if (prev >= 0) {
          indices += (prev.toInt -> (start, i - start))
        }
        start = i
      }
      prev = result.valueAt(i, 1)
      if (i == result.size(1)) {
        indices += (prev.toInt -> (start, i - start + 1))
      }
      i += 1
    }
    indices
  }

  def encodeBBox(priorBox: Tensor[Float], priorVariance: Tensor[Float], gtBox: Tensor[Float],
    enodeBbox: Tensor[Float]): Unit = {
    val px1 = priorBox.valueAt(1)
    val py1 = priorBox.valueAt(2)
    val px2 = priorBox.valueAt(3)
    val py2 = priorBox.valueAt(4)
    val prior = BoundingBox(priorBox.valueAt(1), priorBox.valueAt(2),
      priorBox.valueAt(3), priorBox.valueAt(4))
    val bbox = BoundingBox(gtBox.valueAt(4), gtBox.valueAt(5),
      gtBox.valueAt(6), gtBox.valueAt(7))
    val priorWidth = prior.width()
    val priorHeight = prior.height()
    val bboxWidth = bbox.width()
    val bboxHeight = bbox.height()
    enodeBbox.setValue(1,
      (bbox.centerX() - prior.centerX()) / priorWidth / priorVariance.valueAt(1))
    enodeBbox.setValue(2,
      (bbox.centerY() - prior.centerY()) / priorHeight / priorVariance.valueAt(2))
    enodeBbox.setValue(3, Math.log(bboxWidth / priorWidth).toFloat / priorVariance.valueAt(3))
    enodeBbox.setValue(4, Math.log(bboxHeight / priorHeight).toFloat / priorVariance.valueAt(4))
  }

  def getGroundTruthIndices(result: Tensor[Float]): Map[Int, (Int, Int)] = {
    var indices = Map[Int, (Int, Int)]()
    if (result.nElement() == 0) return indices
    var prev = -1f
    var i = 1
    var start = 1
    if (result.size(1) == 1) {
      indices += (result.valueAt(i, 1).toInt -> (1, 1))
      return indices
    }
    while (i <= result.size(1)) {
      if (prev != result.valueAt(i, 1)) {
        if (prev >= 0) {
          indices += (prev.toInt -> (start, i - start))
        }
        start = i
      }
      prev = result.valueAt(i, 1)
      if (i == result.size(1)) {
        indices += (prev.toInt -> (start, i - start + 1))
      }
      i += 1
    }
    indices
  }

  def getGroundTruths(result: Tensor[Float]): Map[Int, Tensor[Float]] = {
    val indices = getGroundTruthIndices(result).toArray.sortBy(_._1)
    var gtMap = Map[Int, Tensor[Float]]()
    var ind = 0
    val iter = indices.iterator
    while (iter.hasNext) {
      val x = iter.next()
      val gt = result.narrow(1, x._2._1, x._2._2)
      // -1 represent those images without label
      if (gt.size(1) > 1 || gt.valueAt(1, 2) != -1) {
        gtMap += (ind -> gt)
      }
      ind += 1
    }
    gtMap
    //    indices.map(x => x._1 -> result.narrow(1, x._2._1, x._2._2))
  }

  // inplace scale
  def scaleBBox(classBboxes: Tensor[Float], height: Float, width: Float): Unit = {
    if (classBboxes.nElement() == 0) return
    classBboxes.select(2, 1).mul(width)
    classBboxes.select(2, 2).mul(height)
    classBboxes.select(2, 3).mul(width)
    classBboxes.select(2, 4).mul(height)
  }


  val logger = Logger.getLogger(getClass)

  /**
   * Note that the output are stored in input deltas
   * @param boxes (N, 4)
   * @param deltas (N, 4a)
   * @return
   */
  def bboxTransformInv(boxes: Tensor[Float], deltas: Tensor[Float]): Tensor[Float] = {
    if (boxes.size(1) == 0) {
      return boxes
    }
    val output = Tensor[Float]().resizeAs(deltas).copy(deltas)
    require(boxes.size(2) == 4,
      s"boxes size ${boxes.size().mkString(",")} do not satisfy N*4 size")
    require(output.size(2) % 4 == 0,
      s"and deltas size ${output.size().mkString(",")} do not satisfy N*4a size")
    val boxesArr = boxes.storage().array()
    var offset = boxes.storageOffset() - 1
    val rowLength = boxes.stride(1)
    val deltasArr = output.storage().array()
    var i = 0
    val repeat = output.size(2) / boxes.size(2)
    var deltasoffset = output.storageOffset() - 1
    while (i < boxes.size(1)) {
      val x1 = boxesArr(offset)
      val y1 = boxesArr(offset + 1)
      val width = boxesArr(offset + 2) - x1 + 1
      val height = boxesArr(offset + 3) - y1 + 1
      var j = 0
      while (j < repeat) {
        j += 1
        // dx1*width + centerX
        val predCtrX = deltasArr(deltasoffset) * width + x1 + width / 2
        // dy1*height + centerY
        val predCtrY = deltasArr(deltasoffset + 1) * height + y1 + height / 2
        // exp(dx2)*width/2
        val predW = Math.exp(deltasArr(deltasoffset + 2)).toFloat * width / 2
        // exp(dy2)*height/2
        val predH = Math.exp(deltasArr(deltasoffset + 3)).toFloat * height / 2
        deltasArr(deltasoffset) = predCtrX - predW
        deltasArr(deltasoffset + 1) = predCtrY - predH
        deltasArr(deltasoffset + 2) = predCtrX + predW
        deltasArr(deltasoffset + 3) = predCtrY + predH
        deltasoffset += rowLength
      }
      offset += rowLength
      i += 1
    }
    output
  }

  /**
   * Clip boxes to image boundaries.
   * set the score of all boxes with any side smaller than minSize to 0
   * @param boxes N * 4a
   * @param height height of image
   * @param width width of image
   * @param minH min height limit
   * @param minW min width limit
   * @param scores scores for boxes
   * @return the number of boxes kept (score > 0)
   */
  def clipBoxes(boxes: Tensor[Float], height: Float, width: Float, minH: Float = 0,
    minW: Float = 0, scores: Tensor[Float] = null): Int = {
    require(boxes.size(2) % 4 == 0, "boxes should have the shape N*4a")
    val boxesArr = boxes.storage().array()
    var offset = boxes.storageOffset() - 1
    val scoresArr = if (scores != null) scores.storage().array() else null
    var scoreOffset = if (scores != null) scores.storageOffset() - 1 else -1
    var i = 0
    var count = 0
    val h = height - 1
    val w = width - 1
    val repeat = boxes.size(2) / 4
    while (i < boxes.size(1)) {
      var r = 0
      while (r < repeat) {
        boxesArr(offset) = Math.max(Math.min(boxesArr(offset), w), 0)
        boxesArr(offset + 1) = Math.max(Math.min(boxesArr(offset + 1), h), 0)
        boxesArr(offset + 2) = Math.max(Math.min(boxesArr(offset + 2), w), 0)
        boxesArr(offset + 3) = Math.max(Math.min(boxesArr(offset + 3), h), 0)

        if (scores != null) {
          val width = boxesArr(offset + 2) - boxesArr(offset) + 1
          if (width < minW) {
            scoresArr(scoreOffset) = 0
          } else {
            val height = boxesArr(offset + 3) - boxesArr(offset + 1) + 1
            if (height < minH) scoresArr(scoreOffset) = 0
            else count += 1
          }
          scoreOffset += 1
        }
        r += 1
        offset += 4
      }
      i += 1
    }
    count
  }

  /**
   *
   * @param scoresNms N
   * @param bboxNms N * 4
   * @param scoresAll M
   * @param bboxAll M * 4
   * @return
   */
  def bboxVote(scoresNms: Tensor[Float], bboxNms: Tensor[Float],
    scoresAll: Tensor[Float], bboxAll: Tensor[Float],
    areasBuf: Tensor[Float] = null): RoiLabel = {
    var accBox: Tensor[Float] = null
    var accScore = 0f
    var box: Tensor[Float] = null
    val areasAll = if (areasBuf == null) {
      Tensor[Float]
    } else areasBuf
    getAreas(bboxAll, areasAll)
    var i = 1
    while (i <= scoresNms.size(1)) {
      box = bboxNms(i)
      if (accBox == null) {
        accBox = Tensor[Float](4)
      } else {
        accBox.fill(0f)
      }
      accScore = 0f
      var m = 1
      while (m <= scoresAll.size(1)) {
        val boxA = bboxAll(m)
        val iw = Math.min(box.valueAt(3), boxA.valueAt(3)) -
          Math.max(box.valueAt(1), boxA.valueAt(1)) + 1
        val ih = Math.min(box.valueAt(4), boxA.valueAt(4)) -
          Math.max(box.valueAt(2), boxA.valueAt(2)) + 1

        if (iw > 0 && ih > 0) {
          val ua = getArea(box) + areasAll.valueAt(m) - iw * ih
          val ov = iw * ih / ua
          if (ov >= 0.5) {
            accBox.add(scoresAll.valueAt(m), boxA)
            accScore += scoresAll.valueAt(m)
          }
        }
        m += 1
      }
      var x = 1
      while (x <= 4) {
        bboxNms.setValue(i, x, accBox.valueAt(x) / accScore)
        x += 1
      }
      i += 1
    }
    RoiLabel(scoresNms, bboxNms)
  }

  private def getArea(box: Tensor[Float]): Float = {
    require(box.dim() == 1 && box.nElement() >= 4)
    (box.valueAt(3) - box.valueAt(1) + 1) * (box.valueAt(4) - box.valueAt(2) + 1)
  }

  /**
   * get the areas of boxes
   * @param boxes N * 4 tensor
   * @param areas buffer to store the results
   * @return areas array
   */
  def getAreas(boxes: Tensor[Float], areas: Tensor[Float], startInd: Int = 1,
    normalized: Boolean = false): Tensor[Float] = {
    if (boxes.nElement() == 0) return areas
    require(boxes.size(2) >= 4)
    areas.resize(boxes.size(1))
    val boxesArr = boxes.storage().array()
    val offset = boxes.storageOffset() - 1
    val rowLength = boxes.stride(1)
    var i = 0
    var boffset = offset + startInd - 1
    while (i < boxes.size(1)) {
      val x1 = boxesArr(boffset)
      val y1 = boxesArr(boffset + 1)
      val x2 = boxesArr(boffset + 2)
      val y2 = boxesArr(boffset + 3)
      if (normalized) areas.setValue(i + 1, (x2 - x1) * (y2 - y1))
      else areas.setValue(i + 1, (x2 - x1 + 1) * (y2 - y1 + 1))
      boffset += rowLength
      i += 1
    }
    areas
  }

  private def decodeSingleBbox(i: Int, priorBox: Tensor[Float], priorVariance: Tensor[Float],
    isClipBoxes: Boolean, bbox: Tensor[Float], varianceEncodedInTarget: Boolean,
    decodedBoxes: Tensor[Float]): Unit = {
    val x1 = priorBox.valueAt(i, 1)
    val y1 = priorBox.valueAt(i, 2)
    val x2 = priorBox.valueAt(i, 3)
    val y2 = priorBox.valueAt(i, 4)
    val priorWidth = x2 - x1
    require(priorWidth > 0)
    val priorHeight = y2 - y1
    require(priorHeight > 0)
    val pCenterX = (x1 + x2) / 2
    val pCenterY = (y1 + y2) / 2
    var decodeCenterX = 0f
    var decodeCenterY = 0f
    var decodeWidth = 0f
    var decodedHeight = 0f
    if (varianceEncodedInTarget) {
      // variance is encoded in target, we simply need to retore the offset
      // predictions.
      decodeCenterX = bbox.valueAt(i, 1) * priorWidth + pCenterX
      decodeCenterY = bbox.valueAt(i, 2) * priorHeight + pCenterY
      decodeWidth = Math.exp(bbox.valueAt(i, 3)).toFloat * priorWidth
      decodedHeight = Math.exp(bbox.valueAt(i, 4)).toFloat * priorHeight
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decodeCenterX = priorVariance.valueAt(i, 1) * bbox.valueAt(i, 1) * priorWidth + pCenterX
      decodeCenterY = priorVariance.valueAt(i, 2) * bbox.valueAt(i, 2) * priorHeight + pCenterY
      decodeWidth = Math.exp(priorVariance.valueAt(i, 3) * bbox.valueAt(i, 3)).toFloat * priorWidth
      decodedHeight = Math.exp(priorVariance.valueAt(i, 4) * bbox.valueAt(i, 4))
        .toFloat * priorHeight
    }
    decodedBoxes.setValue(i, 1, decodeCenterX - decodeWidth / 2)
    decodedBoxes.setValue(i, 2, decodeCenterY - decodedHeight / 2)
    decodedBoxes.setValue(i, 3, decodeCenterX + decodeWidth / 2)
    decodedBoxes.setValue(i, 4, decodeCenterY + decodedHeight / 2)
    if (isClipBoxes) {
      clipBoxes(decodedBoxes)
    }
  }

  def decodeBoxes(priorBoxes: Tensor[Float], priorVariances: Tensor[Float],
    isClipBoxes: Boolean, bboxes: Tensor[Float],
    varianceEncodedInTarget: Boolean, output: Tensor[Float] = null): Tensor[Float] = {
    require(priorBoxes.size(1) == priorVariances.size(1))
    require(priorBoxes.size(1) == bboxes.size(1))
    val numBboxes = priorBoxes.size(1)
    if (numBboxes > 0) {
      require(priorBoxes.size(2) == 4)
    }
    val decodedBboxes = if (output == null) Tensor[Float](numBboxes, 4)
    else output.resizeAs(priorBoxes)
    var i = 1
    while (i <= numBboxes) {
      decodeSingleBbox(i, priorBoxes,
        priorVariances, isClipBoxes, bboxes, varianceEncodedInTarget, decodedBboxes)
      i += 1
    }
    decodedBboxes
  }

  def clipBoxes(bboxes: Tensor[Float]): Tensor[Float] = {
    bboxes.cmax(0).apply1(x => Math.min(1, x))
  }

  def decodeBboxesAll(allLocPreds: Array[Array[Tensor[Float]]], priorBoxes: Tensor[Float],
    priorVariances: Tensor[Float], nClasses: Int, bgLabel: Int, clipBoxes: Boolean,
    varianceEncodedInTarget: Boolean, shareLocation: Boolean,
    output: Array[Array[Tensor[Float]]] = null)
  : Array[Array[Tensor[Float]]] = {
    val batch = allLocPreds.length
    val allDecodeBboxes = if (output == null) {
      val all = new Array[Array[Tensor[Float]]](batch)
      var i = 0
      while (i < batch) {
        all(i) = new Array[Tensor[Float]](nClasses)
        i += 1
      }
      all
    } else {
      require(output.length == batch)
      output
    }
    var i = 0
    while (i < batch) {
      val decodedBoxes = allDecodeBboxes(i)
      var c = 0
      while (c < nClasses) {
        // Ignore background class.
        if (shareLocation || c != bgLabel) {
          // Something bad happened if there are no predictions for current label.
          if (allLocPreds(i)(c).nElement() == 0) {
            logger.warn(s"Could not find location predictions for label $c")
          }
          val labelLocPreds = allLocPreds(i)(c)
          decodedBoxes(c) = decodeBoxes(priorBoxes, priorVariances, clipBoxes,
            labelLocPreds, varianceEncodedInTarget, labelLocPreds)
        }
        c += 1
      }
      allDecodeBboxes(i) = decodedBoxes
      i += 1
    }
    allDecodeBboxes
  }

  def getLocPredictions(loc: Tensor[Float], numPredsPerClass: Int, numClasses: Int,
    shareLocation: Boolean, locPredsBuf: Array[Array[Tensor[Float]]] = null)
  : Array[Array[Tensor[Float]]] = {
    // the outer array is the batch, each img contains an array of results, grouped by class
    val locPreds = if (locPredsBuf == null) {
      val out = new Array[Array[Tensor[Float]]](loc.size(1))
      var i = 0
      while (i < loc.size(1)) {
        out(i) = new Array[Tensor[Float]](numClasses)
        var c = 0
        while (c < numClasses) {
          out(i)(c) = Tensor[Float](numPredsPerClass, 4)
          c += 1
        }
        i += 1
      }
      out
    } else {
      locPredsBuf
    }
    var i = 0
    val locData = loc.storage().array()
    var locDataOffset = loc.storageOffset() - 1
    while (i < loc.size(1)) {
      val labelBbox = locPreds(i)
      var p = 0
      while (p < numPredsPerClass) {
        val startInd = p * numClasses * 4 + locDataOffset
        var c = 0
        while (c < numClasses) {
          val label = if (shareLocation) labelBbox.length - 1 else c
          val boxData = labelBbox(label).storage().array()
          val boxOffset = p * 4 + labelBbox(label).storageOffset() - 1
          val offset = startInd + c * 4
          boxData(boxOffset) = locData(offset)
          boxData(boxOffset + 1) = locData(offset + 1)
          boxData(boxOffset + 2) = locData(offset + 2)
          boxData(boxOffset + 3) = locData(offset + 3)
          c += 1
        }
        p += 1
      }
      locDataOffset += numPredsPerClass * numClasses * 4
      i += 1
    }
    locPreds
  }

  def getConfidenceScores(conf: Tensor[Float], numPredsPerClass: Int, numClasses: Int,
    confBuf: Array[Array[Tensor[Float]]] = null)
  : Array[Array[Tensor[Float]]] = {
    val confPreds = if (confBuf == null) {
      val out = new Array[Array[Tensor[Float]]](conf.size(1))
      var i = 0
      while (i < conf.size(1)) {
        out(i) = new Array[Tensor[Float]](numClasses)
        var c = 0
        while (c < numClasses) {
          out(i)(c) = Tensor[Float](numPredsPerClass)
          c += 1
        }
        i += 1
      }
      out
    }
    else confBuf
    val confData = conf.storage().array()
    var confDataOffset = conf.storageOffset() - 1
    var i = 0
    while (i < conf.size(1)) {
      val labelScores = confPreds(i)
      var p = 0
      while (p < numPredsPerClass) {
        val startInd = p * numClasses + confDataOffset
        var c = 0
        while (c < numClasses) {
          labelScores(c).setValue(p + 1, confData(startInd + c))
          c += 1
        }
        p += 1
      }
      confDataOffset += numPredsPerClass * numClasses
      i += 1
    }
    confPreds
  }

  def getPriorBboxes(prior: Tensor[Float], nPriors: Int): (Tensor[Float], Tensor[Float]) = {
    val array = prior.storage()
    val aOffset = prior.storageOffset()
    val priorBoxes = Tensor(array, aOffset, Array(nPriors, 4))
    val priorVariances = Tensor(array, aOffset + nPriors * 4, Array(nPriors, 4))
    (priorBoxes, priorVariances)
  }

  def locateBBox(srcBox: BoundingBox, box: BoundingBox, locBox: BoundingBox)
  : Unit = {
    val srcW = srcBox.width()
    val srcH = srcBox.height()
    locBox.x1 = srcBox.x1 + box.x1 * srcW
    locBox.y1 = srcBox.y1 + box.y1 * srcH
    locBox.x2 = srcBox.x1 + box.x2 * srcW
    locBox.y2 = srcBox.y1 + box.y2 * srcH
  }


  def clipBox(box: BoundingBox, clipedBox: BoundingBox): Unit = {
    clipedBox.x1 = Math.max(Math.min(box.x1, 1f), 0f)
    clipedBox.y1 = Math.max(Math.min(box.y1, 1f), 0f)
    clipedBox.x2 = Math.max(Math.min(box.x2, 1f), 0f)
    clipedBox.y2 = Math.max(Math.min(box.y2, 1f), 0f)
  }

  def scaleBox(box: BoundingBox, height: Int, width: Int, scaledBox: BoundingBox): Unit = {
    scaledBox.x1 = box.x1 * width
    scaledBox.y1 = box.y1 * height
    scaledBox.x2 = box.x2 * width
    scaledBox.y2 = box.y2 * height
  }


  /**
   * Project bbox onto the coordinate system defined by src_bbox.
   * @param srcBox
   * @param bbox
   * @param projBox
   * @return
   */
  def projectBbox(srcBox: BoundingBox, bbox: BoundingBox,
    projBox: BoundingBox): Boolean = {
    if (bbox.x1 >= srcBox.x2 || bbox.x2 <= srcBox.x1 ||
      bbox.y1 >= srcBox.y2 || bbox.y2 <= srcBox.y1) {
      return false
    }
    val srcWidth = srcBox.width()
    val srcHeight = srcBox.height()
    projBox.x1 = (bbox.x1 - srcBox.x1) / srcWidth
    projBox.y1 = (bbox.y1 - srcBox.y1) / srcHeight
    projBox.x2 = (bbox.x2 - srcBox.x1) / srcWidth
    projBox.y2 = (bbox.y2 - srcBox.y1) / srcHeight
    clipBox(projBox, projBox)
    if (projBox.area() > 0) true
    else false
  }

  def jaccardOverlap(bbox: BoundingBox, bbox2: BoundingBox): Float = {
    val w = math.min(bbox.x2, bbox2.x2) - math.max(bbox.x1, bbox2.x1)
    if (w < 0) return 0
    val h = math.min(bbox.y2, bbox2.y2) - math.max(bbox.y1, bbox2.y1)
    if (h < 0) return 0
    val overlap = w * h
    overlap / ((bbox.area() + bbox2.area()) - overlap)
  }

  def meetEmitCenterConstraint(srcBox: BoundingBox, bbox: BoundingBox): Boolean = {
    val xCenter = bbox.centerX()
    val yCenter = bbox.centerY()
    if (xCenter >= srcBox.x1 && xCenter <= srcBox.x2 &&
      yCenter >= srcBox.y1 && yCenter <= srcBox.y2) {
      true
    } else {
      false
    }
  }

  def getMaxOverlaps(gtBboxes: Tensor[Float],
    gtAreas: Tensor[Float], gtInds: Array[Int],
    bbox: Tensor[Float], normalized: Boolean = false, overlaps: Tensor[Float] = null,
    gtOffset: Int = 0)
  : (Float, Int) = {
    require(bbox.dim() == 1)
    var maxOverlap = Float.MinValue
    var maxInd = -1
    if (overlaps != null) overlaps.resizeAs(gtAreas)

    if (gtInds != null && gtInds.length > 0) {
      var i = 1
      while (i <= gtInds.length) {
        val r = gtInds(i - 1)
        val ixmin = Math.max(gtBboxes.valueAt(r, 1 + gtOffset), bbox.valueAt(1))
        val iymin = Math.max(gtBboxes.valueAt(r, 2 + gtOffset), bbox.valueAt(2))
        val ixmax = Math.min(gtBboxes.valueAt(r, 3 + gtOffset), bbox.valueAt(3))
        val iymax = Math.min(gtBboxes.valueAt(r, 4 + gtOffset), bbox.valueAt(4))
        val (inter, bbArea) = if (normalized) {
          val inter = Math.max(ixmax - ixmin, 0) * Math.max(iymax - iymin, 0)
          val bbArea = (bbox.valueAt(3) - bbox.valueAt(1)) *
            (bbox.valueAt(4) - bbox.valueAt(2))
          (inter, bbArea)
        } else {
          val inter = Math.max(ixmax - ixmin + 1, 0) * Math.max(iymax - iymin + 1, 0)
          val bbArea = (bbox.valueAt(3) - bbox.valueAt(1) + 1f) *
            (bbox.valueAt(4) - bbox.valueAt(2) + 1f)
          (inter, bbArea)
        }

        val overlap = inter / (gtAreas.valueAt(r) - inter + bbArea)
        if (maxOverlap < overlap) {
          maxOverlap = overlap
          maxInd = i - 1
        }
        if (overlaps != null) overlaps.setValue(i, overlap)
        i += 1
      }
    }
    (maxOverlap, maxInd)
  }

}




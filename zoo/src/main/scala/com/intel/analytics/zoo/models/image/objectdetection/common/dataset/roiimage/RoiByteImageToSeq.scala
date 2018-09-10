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

package com.intel.analytics.zoo.models.image.objectdetection.common.dataset.roiimage

import java.io.File
import java.nio.ByteBuffer
import java.nio.file.Path

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.commons.io.FileUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path => hPath}
import org.apache.hadoop.io.{SequenceFile, Text}
import RoiByteImageToSeq._
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel

import scala.collection.Iterator

class RoiByteImageToSeq(blockSize: Int, baseFileName: Path) extends
  Transformer[ImageFeature, String] {
  private val conf: Configuration = new Configuration
  private var index = 0
  private val preBuffer: ByteBuffer = ByteBuffer.allocate(4 * 2)
  private var classLen: Int = 0
  private var startInd: Int = 0

  override def apply(prev: Iterator[ImageFeature]): Iterator[String] = {
    new Iterator[String] {
      override def hasNext: Boolean = prev.hasNext

      override def next(): String = {
        val fileName = baseFileName + s"_$index.seq"
        val path = new hPath(fileName)
        val writer = SequenceFile.createWriter(conf, SequenceFile.Writer.file(path),
          SequenceFile.Writer.keyClass(classOf[Text]),
          SequenceFile.Writer.valueClass(classOf[Text]))
        var i = 0
        while (i < blockSize && prev.hasNext) {
          val roidb = prev.next()
          val imageInByte = FileUtils.readFileToByteArray(new File(roidb.uri()))
          val target = roidb.getLabel[RoiLabel]
          classLen = if (target != null) target.classes.size(2) * 4 else 0
          preBuffer.putInt(imageInByte.length)
          preBuffer.putInt(classLen)
          val data: Array[Byte] = new Array[Byte](preBuffer.capacity + imageInByte.length +
            classLen * 6 * 4)
          startInd = 0
          System.arraycopy(preBuffer.array, 0, data, startInd, preBuffer.capacity)
          startInd += preBuffer.capacity
          System.arraycopy(imageInByte, 0, data, startInd, imageInByte.length)
          startInd += imageInByte.length
          if (target != null) {
            val cls = tensorToBytes(target.classes)
            System.arraycopy(cls, 0, data, startInd, cls.length)
            startInd += cls.length
            val bbox = tensorToBytes(target.bboxes)
            System.arraycopy(bbox, 0, data, startInd, bbox.length)
          }
          preBuffer.clear
          writer.append(new Text(roidb.uri()), new Text(data))
          i += 1
        }
        writer.close()
        index += 1
        fileName
      }
    }
  }


}

object RoiByteImageToSeq {
  def apply(blockSize: Int, baseFileName: Path): RoiByteImageToSeq =
    new RoiByteImageToSeq(blockSize, baseFileName)

  def tensorToBytes(tensor: Tensor[Float]): Array[Byte] = {
    val boxBuffer = ByteBuffer.allocate(tensor.nElement() * 4)
    var i = 0
    val td = tensor.storage().array()
    while (i < td.length) {
      boxBuffer.putFloat(td(i))
      i += 1
    }
    boxBuffer.array()
  }
}


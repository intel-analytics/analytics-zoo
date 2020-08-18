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

package com.intel.analytics.zoo.serving.engine

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.serving.postprocessing.PostProcessing
import com.intel.analytics.zoo.serving.utils.SerParams
import org.apache.log4j.Logger

object InferenceSupportive {
  val logger = Logger.getLogger(getClass)
  def singleThreadInference(preProcessed: Iterator[(String, Activity)],
                           params: SerParams): Iterator[(String, String)] = {
    val postProcessed = preProcessed.grouped(params.coreNum).flatMap(pathByteBatch => {
      val thisBatchSize = pathByteBatch.size
      (0 until thisBatchSize).toParArray.map(idx => {
        try {
          val t = pathByteBatch(idx)._2
          val result = params.model.doPredict(t)
          val value = PostProcessing(result.toTensor[Float], params.filter)
          (pathByteBatch(idx)._1, value)
        } catch {
          case e:Exception =>
            logger.error(e.getMessage)
            e.printStackTrace()
            logger.error("Your input format is invalid to your model, this batch is skipped")
            null
        }
      }).filter(x => x != null)
    })
    postProcessed
  }
  def multiThreadInference(preProcessed: Iterator[(String, Activity)],
                           params: SerParams): Iterator[(String, String)] = {
    println("Inference new batch ..................")
    val postProcessed = preProcessed.grouped(params.coreNum).flatMap(pathByteBatch => {
      try {
        val thisBatchSize = pathByteBatch.size

        val t1 = System.nanoTime()
        val t = batchInput(pathByteBatch, params)
        val t2 = System.nanoTime()
        println(s"Batch input (copy, resize) time ${(t2 - t1) / 1e9} s")
        /**
         * addSingletonDimension method will modify the
         * original Tensor, thus if reuse of Tensor is needed,
         * have to squeeze it back.
         */
        dimCheck(t, "add", params)
        println(t)
        println(t.getClass)
        val result = params.model.doPredict(t)
        println(result)
        dimCheck(result, "remove", params)
        dimCheck(t, "remove", params)
        val t3 = System.nanoTime()
        println(s"Inference and Dim check time ${(t3 - t2) / 1e9} s")
        val kvResult = if (result.isTensor) {
          (0 until thisBatchSize).toParArray.map(i => {
            val value = PostProcessing(result.toTensor[Float].select(1, i + 1), params.filter)
            (pathByteBatch(i)._1, value)
          })
        } else if (result.isTable) {
          val dataTable = result.toTable
          (0 until thisBatchSize).toParArray.map(i => {
            var value = ""
            dataTable.keySet.foreach(key => {
              value += PostProcessing(dataTable(key).asInstanceOf[Tensor[Float]]
                .select(1, i + 1), params.filter)
            })
            (pathByteBatch(i)._1, value)
          })
        } else {
          throw new Exception("Wrong output format, neither Tensor nor Table.")
        }
        val t4 = System.nanoTime()
        println(s"Post-processing time ${(t4 - t3) / 1e9} s")
        println(s"Inference logic total time ${(t4 - t1) / 1e9} s")
        kvResult
      } catch {
        case e:Exception =>
          logger.error(e.getMessage)
          e.printStackTrace()
          logger.error("Your input format is invalid to your model, this batch is skipped")
          pathByteBatch.toParArray.map(x => (x._1, ""))
      }
    })
    postProcessed.filter(x => x != null)
  }
  def batchInput(seq: Seq[(String, Activity)],
    params: SerParams): Activity = {
    val thisBatchSize = seq.size
    println(s"This batch size is ${thisBatchSize.toString}")

    val inputSample = seq.head._2.toTable
    val kvTuples = inputSample.keySet.map(key => {
      (key, Tensor[Float](params.coreNum +:
        inputSample(key).asInstanceOf[Tensor[Float]].size()))
    }).toList
    val t = T(kvTuples.head, kvTuples.tail: _*)
    // Batch tensor and copy
    (0 until thisBatchSize).toParArray.foreach(i => {
      val dataTable = seq(i)._2.toTable
      t.keySet.foreach(key => {
        t(key).asInstanceOf[Tensor[Float]].select(1, i + 1)
          .copy(dataTable(key).asInstanceOf[Tensor[Float]])
      })
    })
    // Resize and specific control
    t.keySet.foreach(key => {
      val singleTensorSize = inputSample(key).asInstanceOf[Tensor[Float]].size()
      var newSize = Array(thisBatchSize)
      for (elem <- singleTensorSize) {
        newSize = newSize :+ elem
      }
      t(key).asInstanceOf[Tensor[Float]].resize(newSize)
    })
    if (t.keySet.size == 1) {
      t.keySet.foreach(key => {
        return t(key).asInstanceOf[Tensor[Float]]
      })
    }
    t
  }
  def dimCheck(input: Activity, op: String, params: SerParams): Activity = {
    if (params.modelType == "openvino") {
      if (input.isTensor) {
        op match {
          case "add" => input.asInstanceOf[Tensor[Float]].addSingletonDimension()
          case _ => input.asInstanceOf[Tensor[Float]].squeeze(1)
        }
      }
      else if (input.isTable) {
        val dataTable = input.toTable
        op match {
          case "add" => dataTable.keySet.foreach(key => {
            dataTable(key).asInstanceOf[Tensor[Float]].addSingletonDimension()
          })
          case _ => dataTable.keySet.foreach(key => {
            dataTable(key).asInstanceOf[Tensor[Float]].squeeze(1)
          })
        }
      }
    }
    input
  }
}

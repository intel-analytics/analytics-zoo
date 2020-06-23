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

package com.intel.analytics.zoo.pipeline.inference

import java.io.{File, IOException}
import java.nio.file.{Files, Paths}
import java.util.{ArrayList, Arrays, List => JList}

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.zoo.pipeline.api.net.{NetUtils, RegistryMap, SerializationHolder}
import com.intel.analytics.zoo.pipeline.inference.DeviceType.DeviceTypeEnumVal
import com.intel.analytics.zoo.pipeline.inference.OpenVINOModel.OpenVINOModelHolder
import org.apache.commons.io.FileUtils
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

class OpenVINOModel(var modelHolder: OpenVINOModelHolder,
                    var isInt8: Boolean,
                    var batchSize: Int = -1,
                    var deviceType: DeviceTypeEnumVal = DeviceType.CPU)
  extends AbstractModel with InferenceSupportive with Serializable {

  private var isRelease: Boolean = false

  @transient
  private lazy val supportive: OpenVinoInferenceSupportive = {
    OpenVINOModel.logger.info("Prepare OpenVinoInferenceSupportive")
    OpenVinoInferenceSupportive.forceLoad()
    new OpenVinoInferenceSupportive()
  }

  @transient
  private lazy val executableNetworkReference: Long = {
    OpenVINOModel.logger.info("Lazy loading OpenVINO model")
    if (isInt8) {
      OpenVINOModel.logger.info(s"Load int8 model")
      supportive.loadOpenVinoIRInt8(modelHolder.modelPath,
        modelHolder.weightPath,
        deviceType.value, batchSize)
    } else {
      OpenVINOModel.logger.info(s"Load fp32 model")
      supportive.loadOpenVinoIR(modelHolder.modelPath,
        modelHolder.weightPath,
        deviceType.value, batchSize)
    }
  }

  override def predict(inputs: JList[JList[JTensor]]): JList[JList[JTensor]] = {
    val outputs = new ArrayList[JList[JTensor]]()
    inputs.asScala.map(input => {
      val tensor = input.get(0)
      val output = if (isInt8) {
        supportive.predictInt8(executableNetworkReference,
          tensor.getData, tensor.getShape)
      } else {
        supportive.predict(executableNetworkReference,
          tensor.getData, tensor.getShape)
      }
      outputs.add(Arrays.asList({
        output
      }))
    })
    outputs
  }

  override def predict(inputActivity: Activity): Activity = {
    val (inputList, batchSize) = inputActivity.isTable match {
      case true =>
        val inputTable = inputActivity.toTable
        val batchSize = inputTable.length()
        (transferBatchTableToJListOfJListOfJTensor(inputTable, batchSize), batchSize)
      case false =>
        val inputTensor = inputActivity.toTensor[Float]
        val batchSize = inputTensor.size(1)
        (transferBatchTensorToJListOfJListOfJTensor(inputTensor, batchSize), batchSize)
    }
    val outputs = predict(inputList)
    transferListOfActivityToActivityOfBatch(outputs, batchSize)
  }

  override def copy(num: Int): Array[AbstractModel] = Array(this)

  override def release(): Unit = {
    isReleased match {
      case true =>
      case false =>
        supportive.releaseOpenVINOIR(executableNetworkReference)
        // Remove Temp model files
        if (modelHolder.modelPath.contains("ZOOTEMPOpenVINO")) {
          FileUtils.deleteQuietly(new File(modelHolder.modelPath))
          FileUtils.deleteQuietly(new File(modelHolder.weightPath))
        }
        isRelease = true
    }
  }

  override def isReleased(): Boolean = {
    isRelease
  }
}

object OpenVINOModel {

  val logger = LoggerFactory.getLogger(getClass)

  @transient
  private lazy val inDriver = NetUtils.isDriver

  class OpenVINOModelHolder(@transient var modelPath: String,
                            @transient var weightPath: String)
    extends SerializationHolder {

    override def writeInternal(out: CommonOutputStream): Unit = {
      logger.debug("Write OpenVINO model into stream")
      if (inDriver) {
        // Temp Model Bytes
        val localModelBytes = Files.readAllBytes(Paths.get(modelPath))
        // Temp Weight Bytes
        val localWeightBytes = Files.readAllBytes(Paths.get(weightPath))
        out.writeInt(localModelBytes.length)
        timing(s"writing " +
          s"${localModelBytes.length / 1024 / 1024}Mb OpenVINO model to stream") {
          out.write(localModelBytes)
        }
        out.writeInt(localWeightBytes.length)
        timing(s"writing " +
          s"${localWeightBytes.length / 1024 / 1024}Mb OpenVINO weight to stream") {
          out.write(localWeightBytes)
        }
      } else {
        out.writeInt(0)
      }
    }

    override def readInternal(in: CommonInputStream): Unit = {
      val modelLen = in.readInt()
      logger.debug("Read OpenVINO model from stream")
      assert(modelLen >= 0, "OpenVINO model length should be an non-negative integer")
      // Temp Model Bytes
      val localModelBytes = new Array[Byte](modelLen)
      timing("reading OpenVINO model from stream") {
        var numOfBytes = 0
        while (numOfBytes < modelLen) {
          val read = in.read(localModelBytes, numOfBytes, modelLen - numOfBytes)
          numOfBytes += read
        }
      }
      val weightLen = in.readInt()
      assert(weightLen >= 0, "OpenVINO weight length should be an non-negative integer")
      // Temp Weight Bytes
      val localWeightBytes = new Array[Byte](weightLen)
      timing("reading OpenVINO weight from stream") {
        var numOfBytes = 0
        while (numOfBytes < weightLen) {
          val read = in.read(localWeightBytes, numOfBytes, weightLen - numOfBytes)
          numOfBytes += read
        }
      }
      val modelFile = File.createTempFile("ZOOTEMPOpenVINO", ".xml")
      Files.write(Paths.get(modelFile.toURI), localModelBytes)
      val weightFile = File.createTempFile("ZOOTEMPOpenVINO", ".bin")
      Files.write(Paths.get(weightFile.toURI), localWeightBytes)
      this.modelPath = modelFile.getAbsolutePath
      this.weightPath = weightFile.getAbsolutePath
    }
  }

  def apply(modelHolder: OpenVINOModelHolder, isInt8: Boolean): OpenVINOModel = {
    new OpenVINOModel(modelHolder, isInt8)
  }

  def apply(modelHolder: OpenVINOModelHolder, isInt8: Boolean, batchSize: Int): OpenVINOModel = {
    new OpenVINOModel(modelHolder, isInt8, batchSize)
  }
}

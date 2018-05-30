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

package com.intel.analytics.zoo.common

import java.io._

import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasModel
import com.intel.analytics.bigdl.utils.File
import com.intel.analytics.zoo.common.NNContext.getClass
import com.intel.analytics.zoo.pipeline.api.keras.models.KerasNet
import com.intel.analytics.zoo.pipeline.api.net.GraphNet
import org.apache.commons.io.filefilter.WildcardFileFilter
import org.apache.log4j.Logger
import java.util.{List => JList}
import scala.collection.JavaConverters._

import scala.collection.mutable.ArrayBuffer

private[zoo] object Utils {

  def getSubModules[T](module: AbstractModule[Activity, Activity, T]):
  JList[AbstractModule[Activity, Activity, T]] = {
    module match {
      case m: KerasNet[T] =>
        m.getSubModules().asJava
      case m: GraphNet[T] =>
        m.getSubModules().asJava
      case m: Container[Activity, Activity, T] =>
        m.modules.asJava
      case _ =>
        throw new IllegalArgumentException(s"module $module does not have submodules")
    }
  }

  def getFlattenSubModules[T](module: AbstractModule[Activity, Activity, T],
                              includeContainer: Boolean)
  : JList[AbstractModule[Activity, Activity, T]] = {
    val result = ArrayBuffer[AbstractModule[Activity, Activity, T]]()
    doGetFlattenModules(module, includeContainer, result)
    result.toList.asJava
  }

  // TODO: refactor Container and KerasLayer to simplify this logic
  private def hasSubModules[T](module: AbstractModule[Activity, Activity, T]) = {
    module match {
      case km: KerasModel[T] => true
      case c: Container[_, _, _] => true
      case k: KerasNet[T] => true
      case _ => false
    }
  }

  private def doGetFlattenModules[T](
      module: AbstractModule[Activity, Activity, T],
      includeContainer: Boolean,
      result: ArrayBuffer[AbstractModule[Activity, Activity, T]]): Unit = {
    getSubModules(module).asScala.foreach {m =>
      if (hasSubModules(m)) {
        doGetFlattenModules(m.asInstanceOf[Container[Activity, Activity, T]],
          includeContainer,
          result)
      } else {
        result.append(m)
      }
    }
    if (includeContainer) {
      result.append(module)
    }
  }

  private val logger = Logger.getLogger(getClass)

  def listLocalFiles(path: String): Array[File] = {
    val files = new ArrayBuffer[File]()
    listFiles(path, files)
    files.toArray
  }

  def listFiles(path: String, files: ArrayBuffer[File]): Unit = {
    val file = new File(path)
    if (file.isDirectory) {
      file.listFiles().foreach(x => listFiles(x.getAbsolutePath, files))
    } else if (file.isFile) {
      files.append(file)
    } else {
      val filter = new WildcardFileFilter(file.getName)
      file.getParentFile.listFiles(new FilenameFilter {
        override def accept(dir: File, name: String): Boolean = filter.accept(dir, name)
      }).foreach(x => listFiles(x.getAbsolutePath, files))
    }
  }

  def saveBytes(bytes: Array[Byte], fileName: String, isOverwrite: Boolean = false) : Unit = {
    File.saveBytes(bytes, fileName, isOverwrite)
  }

  def logUsageErrorAndThrowException(errMessage: String, cause: Throwable = null): Unit = {
    logger.error(s"********************************Usage Error****************************\n"
      + errMessage)
    throw new AnalyticsZooException(errMessage, cause)
  }

}

class AnalyticsZooException(message: String, cause: Throwable)
  extends Exception(message, cause) {

  def this(message: String) = this(message, null)
}


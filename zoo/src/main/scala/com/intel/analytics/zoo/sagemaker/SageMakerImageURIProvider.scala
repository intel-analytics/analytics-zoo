/*
 * Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *   http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package com.intel.analytics.zoo.sagemaker

import com.amazonaws.regions.Regions

private[sagemaker] object SageMakerImageURIProvider {

  def getImage(region: String, regionAccountMap: Map[String, String],
               algorithmName: String, algorithmTag: String): String = {
    val account = regionAccountMap.get(region)
    account match {
      case None => throw new RuntimeException(s"The region $region is not supported." +
        s"Supported Regions: ${regionAccountMap.keys.mkString(", ")}")
      case _ => s"${account.get}.dkr.ecr.${region}.amazonaws.com/${algorithmName}:${algorithmTag}"
    }
  }
}

private[sagemaker] object SagerMakerRegionAccountMaps {
  // For KMeans, PCA, Linear Learner, FactorizationMachines
  val AlgorithmsAccountMap: Map[String, String] = Map(
    Regions.US_WEST_2.getName -> "785726246675"
  )

}


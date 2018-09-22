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

import com.amazonaws.regions.DefaultAwsRegionProviderChain
import com.amazonaws.services.s3.{AmazonS3, AmazonS3ClientBuilder}
import com.amazonaws.services.securitytoken.{AWSSecurityTokenService, AWSSecurityTokenServiceClientBuilder}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType
import com.amazonaws.services.sagemaker.{AmazonSageMaker, AmazonSageMakerClientBuilder}
import com.amazonaws.services.sagemaker.model.{S3DataDistribution, TrainingInputMode}
import com.amazonaws.services.sagemaker.sparksdk._
import com.amazonaws.services.sagemaker.sparksdk.EndpointCreationPolicy.EndpointCreationPolicy
import com.amazonaws.services.sagemaker.sparksdk.transformation.{RequestRowSerializer, ResponseRowDeserializer}
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.{KMeansProtobufResponseRowDeserializer, XGBoostCSVRowDeserializer}
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.{ProtobufRequestRowSerializer, UnlabeledCSVRequestRowSerializer}
import com.intel.analytics.bigdl.optim.{OptimMethod, SGD, Trigger}
import com.intel.analytics.zoo.pipeline.nnframes.{NNParams, TrainingParams}
import org.apache.spark.ml.param.{IntParam, Param, ParamValidators, Params}
import org.apache.spark.ml.param.shared.HasLabelCol

/**
  * Common params for [[ZooSageMakerEstimator]] with accessors
  */
private[algorithms] trait ZooParams extends Params {

  /**
    * The dimension of the input vectors.
    */
  val featureSize : IntArrayParam = new IntArrayParam(this, "featureSize",
    "The size of the input vectors. Must be > 0.", ParamValidators.arrayLengthGt(0))
  def getFeatureSize: Array[Int] = $(featureSize)

  /**
    * optimization method to be used. BigDL supports many optimization methods like Adam,
    * SGD and LBFGS. Refer to package com.intel.analytics.bigdl.optim for all the options.
    * Default: SGD
    */
  final val optimMethodStr = new Param[String](this, "optimMethod", "optimMethod")

  def getOptimMethodStr: String = $(optimMethodStr)
}

object ZooSageMakerEstimator {
  val algorithmName = "zoo"
  val algorithmTag = "1"
  val regionAccountMap = SagerMakerRegionAccountMaps.AlgorithmsAccountMap
}

/**
  * A [[SageMakerEstimator]] that runs a K-Means Clustering training job on Amazon SageMaker upon a
  * call to fit() on a DataFrame and returns a [[SageMakerModel]]
  * that can be used to transform a DataFrame using the hosted K-Means model. K-Means Clustering
  * is useful for grouping similar examples in your dataset.
  *
  * Amazon SageMaker K-Means clustering trains on RecordIO-encoded Amazon Record protobuf data.
  * SageMaker Spark writes a DataFrame to S3 by selecting a column of Vectors named "features"
  * and, if present, a column of Doubles named "label". These names are configurable by passing
  * a map with entries in trainingSparkDataFormatOptions with key "labelColumnName" or
  * "featuresColumnName", with values corresponding to the desired label and features columns.
  *
  * For inference, the SageMakerModel returned by fit() by the KMeansSageMakerEstimator
  * uses [[ProtobufRequestRowSerializer]] to serialize Rows into
  * RecordIO-encoded Amazon Record protobuf messages for inference, by default selecting
  * the column named "features" expected to contain a Vector of Doubles.
  *
  * Inferences made against an Endpoint hosting a K-Means model contain
  * a "closest_cluster" field and a "distance_to_cluster" field, both
  * appended to the input DataFrame as columns of Double.
  *
  * @param sagemakerRole The SageMaker TrainingJob and Hosting IAM Role. Used by a SageMaker to
  *                      access S3 and ECR resources. SageMaker hosted Endpoints instances
  *                      launched by this Estimator run with this role.
  * @param trainingInstanceType The SageMaker TrainingJob Instance Type to use
  * @param trainingInstanceCount The number of instances of instanceType to run an
  *                              SageMaker Training Job with
  * @param endpointInstanceType The SageMaker Endpoint Confing instance type
  * @param endpointInitialInstanceCount The SageMaker Endpoint Config minimum number of instances
  *                                     that can be used to host modelImage
  * @param requestRowSerializer Serializes Spark DataFrame [[Row]]s for transformation by Models
  *                             built from this Estimator.
  * @param responseRowDeserializer Deserializes an Endpoint response into a series of [[Row]]s.
  * @param trainingInputS3DataPath An S3 location to upload SageMaker Training Job input data to.
  * @param trainingOutputS3DataPath An S3 location for SageMaker to store Training Job output
  *                                 data to.
  * @param trainingInstanceVolumeSizeInGB The EBS volume size in gigabytes of each instance.
  * @param trainingProjectedColumns The columns to project from the Dataset being fit before
  *                                 training. If an Optional.empty is passed then no specific
  *                                 projection will occur and all columns will be serialized.
  * @param trainingChannelName The SageMaker Channel name to input serialized Dataset fit input to
  * @param trainingContentType The MIME type of the training data.
  * @param trainingS3DataDistribution The SageMaker Training Job S3 data distribution scheme.
  * @param trainingSparkDataFormat The Spark Data Format name used to serialize the Dataset being
  *                                fit for input to SageMaker.
  * @param trainingSparkDataFormatOptions The Spark Data Format Options used during serialization of
  *                                       the Dataset being fit.
  * @param trainingInputMode The SageMaker Training Job Channel input mode.
  * @param trainingCompressionCodec The type of compression to use when serializing the Dataset
  *                                 being fit for input to SageMaker.
  * @param trainingMaxRuntimeInSeconds A SageMaker Training Job Termination Condition
  *                                    MaxRuntimeInHours.
  * @param trainingKmsKeyId A KMS key ID for the Output Data Source
  * @param modelEnvironmentVariables The environment variables that SageMaker will set on the model
  *                                  container during execution.
  * @param endpointCreationPolicy Defines how a SageMaker Endpoint referenced by a
  *                               SageMakerModel is created.
  * @param sagemakerClient Amazon SageMaker client. Used to send CreateTrainingJob, CreateModel,
  *                        and CreateEndpoint requests.
  * @param region The region in which to run the algorithm. If not specified, gets the region from
  *               the DefaultAwsRegionProviderChain.
  * @param s3Client AmazonS3. Used to create a bucket for staging SageMaker Training Job input
  *                 and/or output if either are set to S3AutoCreatePath.
  * @param stsClient AmazonSTS. Used to resolve the account number when creating staging
  *                  input / output buckets.
  * @param modelPrependInputRowsToTransformationRows Whether the transformation result on Models
  *        built by this Estimator should also include the input Rows. If true, each output Row
  *        is formed by a concatenation of the input Row with the corresponding Row produced by
  *        SageMaker Endpoint invocation, produced by responseRowDeserializer.
  *        If false, each output Row is just taken from responseRowDeserializer.
  * @param deleteStagingDataAfterTraining Whether to remove the training data on s3 after training
  *                                       is complete or failed.
  * @param namePolicyFactory The [[NamePolicyFactory]] to use when naming SageMaker entities
  *        created during fit
  * @param uid The unique identifier of this Estimator. Used to represent this stage in Spark
  *            ML pipelines.
  */
class ZooSageMakerEstimator(
      override val sagemakerRole : IAMRoleResource = IAMRoleFromConfig(),
      override val trainingInstanceType : String,
      override val trainingInstanceCount : Int,
      override val endpointInstanceType : String,
      override val endpointInitialInstanceCount : Int,
      override val requestRowSerializer : RequestRowSerializer =
        new UnlabeledCSVRequestRowSerializer(),
      override val responseRowDeserializer : ResponseRowDeserializer =
        new XGBoostCSVRowDeserializer(),
      override val trainingInputS3DataPath : S3Resource = S3AutoCreatePath(),
      override val trainingOutputS3DataPath : S3Resource = S3AutoCreatePath(),
      override val trainingInstanceVolumeSizeInGB : Int = 1024,
      override val trainingProjectedColumns : Option[List[String]] = None,
      override val trainingChannelName : String = "train",
      override val trainingContentType: Option[String] = None,
      override val trainingS3DataDistribution : String = S3DataDistribution.ShardedByS3Key.toString,
      override val trainingSparkDataFormat : String = "parquet",
      override val trainingSparkDataFormatOptions : Map[String, String] = Map(),
      override val trainingInputMode : String = TrainingInputMode.File.toString,
      override val trainingCompressionCodec : Option[String] = None,
      override val trainingMaxRuntimeInSeconds : Int = 24 * 60 * 60,
      override val trainingKmsKeyId : Option[String] = None,
      override val modelEnvironmentVariables : Map[String, String] = Map(),
      override val endpointCreationPolicy : EndpointCreationPolicy =
        EndpointCreationPolicy.CREATE_ON_CONSTRUCT,
      override val sagemakerClient : AmazonSageMaker
        = AmazonSageMakerClientBuilder.defaultClient,
      val region : Option[String] = None,
      override val s3Client : AmazonS3 = AmazonS3ClientBuilder.defaultClient(),
      override val stsClient : AWSSecurityTokenService =
        AWSSecurityTokenServiceClientBuilder.defaultClient(),
      override val modelPrependInputRowsToTransformationRows : Boolean = true,
      override val deleteStagingDataAfterTraining : Boolean = true,
      override val namePolicyFactory : NamePolicyFactory = new RandomNamePolicyFactory(),
      override val uid : String = Identifiable.randomUID("zoosagemaker"))
  extends SageMakerEstimator(
    trainingImage = SageMakerImageURIProvider.getImage(
      region.getOrElse(new DefaultAwsRegionProviderChain().getRegion),
      ZooSageMakerEstimator.regionAccountMap,
      ZooSageMakerEstimator.algorithmName, ZooSageMakerEstimator.algorithmTag),
    modelImage = SageMakerImageURIProvider.getImage(
      region.getOrElse(new DefaultAwsRegionProviderChain().getRegion),
      ZooSageMakerEstimator.regionAccountMap,
      ZooSageMakerEstimator.algorithmName, ZooSageMakerEstimator.algorithmTag),
    sagemakerRole,
    trainingInstanceType,
    trainingInstanceCount,
    endpointInstanceType,
    endpointInitialInstanceCount,
    requestRowSerializer,
    responseRowDeserializer,
    trainingInputS3DataPath,
    trainingOutputS3DataPath,
    trainingInstanceVolumeSizeInGB,
    trainingProjectedColumns,
    trainingChannelName,
    trainingContentType,
    trainingS3DataDistribution,
    trainingSparkDataFormat,
    trainingSparkDataFormatOptions,
    trainingInputMode,
    trainingCompressionCodec,
    trainingMaxRuntimeInSeconds,
    trainingKmsKeyId,
    modelEnvironmentVariables,
    endpointCreationPolicy,
    sagemakerClient,
    s3Client,
    stsClient,
    modelPrependInputRowsToTransformationRows,
    deleteStagingDataAfterTraining,
    namePolicyFactory,
    uid) with ZooParams with TrainingParams[Float] with NNParams[Float] {

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

//  def setLabelCol(labelColName : String) : this.type = set(labelCol, labelColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setBatchSize(value: Int): this.type = set(batchSize, value)

  def setEndWhen(trigger: Trigger): this.type = set(endWhen, trigger)

  def setLearningRate(value: Double): this.type = set(learningRate, value)
  setDefault(learningRate -> 1e-3)

  def setLearningRateDecay(value: Double): this.type = set(learningRateDecay, value)
  setDefault(learningRateDecay -> 0.0)

  def setMaxEpoch(value: Int): this.type = set(maxEpoch, value)
  setDefault(maxEpoch -> 50)

  def setOptimMethod(value: String): this.type = set(optimMethodStr, value)
  setDefault(optimMethodStr, "SGD")

  def setConstantGradientClipping(min: Float, max: Float): this.type = {
    set(constantGradientClippingParams, (min, max))
  }

  def setGradientClippingByL2Norm(clipNorm: Float): this.type = {
    set(l2GradientClippingParams, clipNorm)
  }

  def setCachingSample(value: Boolean): this.type = {
    set(cachingSample, value)
  }
  setDefault(cachingSample, true)


  // Check whether required hyper-parameters are set
  override def transformSchema(schema: StructType): StructType = {
    super.transformSchema(schema)
  }

}

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

package com.intel.analytics.zoo.serving


import com.intel.analytics.zoo.pipeline.inference.{InferenceModel, InferenceSummary}
import com.intel.analytics.zoo.serving.utils._
import com.intel.analytics.zoo.serving.engine.InferenceSupportive
import com.intel.analytics.zoo.serving.engine.ServingReceiver
import com.intel.analytics.zoo.serving.pipeline.{RedisIO, RedisUtils}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.streaming.{Duration, StreamingContext}
import redis.clients.jedis.{Jedis, JedisPool}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Failure, Success}
import org.apache.spark.util.LongAccumulator
import com.intel.analytics.zoo.serving.utils.SerParams
import org.apache.spark.storage.StorageLevel

object SparkStreamingClusterServing {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)


  case class Record(uri: String, value: String)

  def main(args: Array[String]): Unit = {

    val helper = new ClusterServingHelper()
    helper.initArgs()
    helper.initContext()

    /**
     * Variables need to be serialized are listed below
     * Take them from helper in advance for later execution
     */

    val modelType = helper.modelType
    val blasFlag = helper.blasFlag

    /**
     * chwFlag is to set image input of CHW or HWC
     * if true, the format is CHW
     * else, the format is HWC
     *
     * Note that currently CHW is commonly used
     * and HWC is often used in Tensorflow models
     */

    val logger = helper.logger

    /**
     * For cut input stream, this variable is to avoid
     * unsafe cut of input stream, if cut current batch
     * there is an interval between get and cut, this would
     * affect the result of correctness, some new data might be cut
     */

//    var model: InferenceModel = null
//    var bcModel: Broadcast[InferenceModel] = null

//    model = helper.loadInferenceModel()
//    bcModel = helper.sc.broadcast(model)



    val spark = helper.getSparkSession()

    logger.info(s"connected to redis " +
      s"${spark.conf.get("spark.redis.host")}:${spark.conf.get("spark.redis.port")}")



    var totalCnt: Int = 0
    var timeStamp: Int = 0

    // redis stream control

    val inputThreshold = 0.6 * 0.8
    val cutRatio = 0.5

    val acc = new LongAccumulator()
    helper.sc.register(acc)



    val localSerParams = new SerParams(helper)
    localSerParams.model.setInferenceSummary(
      InferenceSummary("./TensorboardEventLogs", helper.dateTime + "-ClusterServing"))
    val bcSer = helper.sc.broadcast(localSerParams)

    val jedis = new Jedis(localSerParams.redisHost, localSerParams.redisPort)
    val ssc = new StreamingContext(spark.sparkContext, new Duration(50))


    val receiver = new ServingReceiver(localSerParams.redisHost, localSerParams.redisPort)
    val images = ssc.receiverStream(receiver)

    images.foreachRDD{ m =>
      /**
       * This is reserved for future dynamic loading model
       */
      m.persist(StorageLevel.MEMORY_AND_DISK)

      if (!m.isEmpty) {

        val microBatchStart = System.nanoTime()
        val x = m.coalesce(1)
        acc.reset()

        RedisUtils.checkMemory(jedis, inputThreshold, cutRatio)

        /**
         * The streaming may be triggered somehow and it is possible
         * to get an empty batch
         *
         * If the batch is not empty, start preprocessing and predict here
         */

        val preProcessed = x.mapPartitions(it => {
          val serParams = bcSer.value
          val pre = new PreProcessing(serParams)
          it.grouped(serParams.coreNum).flatMap(itemBatch => {
            acc.add(itemBatch.size)
            itemBatch.indices.toParArray.map(i => {
              val uri = itemBatch(i)._1
              val tensor = pre.decodeArrowBase64(itemBatch(i)._2)
              (uri, tensor)
            })
          })
        })

        /**
         * Engine type controlling, for different engine type,
         * different partitioning and batching scheduling is used
         */
        val postProcessed = if (blasFlag) {
          /**
           * In BLAS mode, every model could predict only using
           * a single thread, besides, batch size usually is not
           * over 64 in serving to achieve good latency. Thus, no
           * batching is required if the machine has over about 30 cores.           *
           */
          preProcessed.mapPartitions(it => {
            val t1 = System.nanoTime()
            val serParams = bcSer.value
            println(s"Take Broadcasted model time ${(System.nanoTime() - t1) / 1e9} s")
            it.grouped(serParams.coreNum).flatMap(itemBatch => {
              InferenceSupportive.multiThreadInference(itemBatch.toIterator, serParams)
            })
          })


        } else {
          /**
           * In Normal mode, every model will use multiple thread to
           * achieve best latency. Thus, we only use a single model to
           * do sequential predict, maximizing the latency performance
           * and minimizing the memory usage.
           */
          preProcessed.mapPartitions(it => {
            val t1 = System.nanoTime()
            val serParams = bcSer.value
            println(s"Take Broadcasted model time ${(System.nanoTime() - t1) / 1e9} s")
            it.grouped(serParams.coreNum).flatMap(itemBatch => {
              InferenceSupportive.multiThreadInference(itemBatch.toIterator, serParams)
            })
          })
        }
        postProcessed.mapPartitions(it => {
          val serParams = bcSer.value
          val jedis = new Jedis(serParams.redisHost, serParams.redisPort)
          val ppl = jedis.pipelined()
          it.foreach(v => RedisIO.writeHashMap(ppl, v._1, v._2))
          ppl.sync()
          jedis.close()
          Iterator(None)
        }).count()



        /**
         * Count the statistical data and write to summary
         */
        val microBatchEnd = System.nanoTime()
        println(s"Currently recs in redis: ${jedis.xlen(Conventions.SERVING_STREAM_NAME)}")
        val microBatchLatency = (microBatchEnd - microBatchStart) / 1e9
        val microBatchThroughPut = (acc.value / microBatchLatency).toFloat
        logger.info(s"Inferece end. Input size ${acc.value}. " +
          s"Latency $microBatchLatency, Throughput $microBatchThroughPut")

        totalCnt += acc.value.toInt

        val lastTimeStamp = timeStamp
        timeStamp += microBatchLatency.toInt + 1

        AsyncUtils.writeServingSummay(localSerParams.model,
          lastTimeStamp, timeStamp, totalCnt, microBatchThroughPut)

        .onComplete{
          case Success(_) => None

          case Failure(exception) => logger.info(s"$exception, " +
            s"write summary fails, please check.")
        }

      }

      m.unpersist()
    }

    /**
     * Start the streaming, and listen to stop signal
     * The stop signal will be listened by another thread
     * with interval of 1 second.
     */
    ssc.start()

    ClusterServingManager.listenTermination(helper, ssc)

    ssc.awaitTermination()

    assert(spark.streams.active.isEmpty)
    System.exit(0)
  }
}


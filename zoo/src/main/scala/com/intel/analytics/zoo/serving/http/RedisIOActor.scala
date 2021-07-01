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

package com.intel.analytics.zoo.serving.http

import java.util
import java.util.{HashMap, UUID}

import akka.actor.{Actor, ActorRef}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.zoo.serving.serialization.{ArrowDeserializer, StreamSerializer}
import com.intel.analytics.zoo.serving.pipeline.RedisUtils
import com.intel.analytics.zoo.serving.utils.Conventions
import org.slf4j.LoggerFactory
import redis.clients.jedis.JedisPool
import org.apache.log4j.{Level, Logger}

import scala.collection.JavaConverters._
import scala.collection.mutable

class RedisIOActor(redisOutputQueue: String = Conventions.RESULT_PREFIX +
  Conventions.SERVING_STREAM_DEFAULT_NAME + ":",
                   redisInputQueue: String = "serving_stream",
                   jedisPool: JedisPool = null,
                   isSilent: Boolean = false) extends Actor with Supportive {
  val levelLogger = Logger.getLogger(getClass)
  if (isSilent){
    levelLogger.setLevel(Level.ERROR)
  }
  val jedis = if (jedisPool == null) {
    RedisUtils.getRedisClient(new JedisPool())
  } else {
    RedisUtils.getRedisClient(jedisPool)
  }
  var requestMap = mutable.Map[String, ActorRef]()

  override def receive: Receive = {
    case message: DataInputMessage =>
      silent(s"${self.path.name} input message process")() {
        levelLogger.info(s"${System.currentTimeMillis()} Input enqueue ${message.id} at time ")
        enqueue(redisInputQueue, message)

        requestMap += (Conventions.RESULT_PREFIX +
          Conventions.SERVING_STREAM_DEFAULT_NAME + ":" + message.id -> sender())
      }
    case message: DequeueMessage =>
      if (!requestMap.isEmpty) {
        dequeue(redisOutputQueue).foreach(result => {
          levelLogger.info(s"${System.currentTimeMillis()} Get redis result at time ")
          val queryOption = requestMap.get(result._1)
          if (queryOption != None) {
            val queryResult = result._2.asScala
            queryOption.get ! ModelOutputMessage(queryResult)
            requestMap -= result._1
            levelLogger.info(s"${System.currentTimeMillis()} Send ${result._1} back at time ")
          }

        })
      }

  }
  def enqueue(queue: String, input: DataInputMessage): Unit = {
    if (isSilent){
      silent(s"${self.path.name} put request to redis")(FrontEndApp.putRedisTimer) {
        val hash = new HashMap[String, String]()
        //      val bytes = StreamSerializer.objToBytes(input.inputs)
        //      val b64 = java.util.Base64.getEncoder.encodeToString(bytes)
        hash.put("uri", input.id)
        hash.put("data", input.inputs)
        hash.put("serde", "stream")
        jedis.xadd(queue, null, hash)
      }
    }else {
      timing(s"${self.path.name} put request to redis")(FrontEndApp.putRedisTimer) {
        val hash = new HashMap[String, String]()
        //      val bytes = StreamSerializer.objToBytes(input.inputs)
        //      val b64 = java.util.Base64.getEncoder.encodeToString(bytes)
        hash.put("uri", input.id)
        hash.put("data", input.inputs)
        hash.put("serde", "stream")
        jedis.xadd(queue, null, hash)
      }
    }
  }
  def dequeue(queue: String): mutable.Set[(String, util.Map[String, String])] = {
    val resultSet = jedis.keys(s"${queue}*")
    val res = resultSet.asScala.map(key => {
      (key, jedis.hgetAll(key))

    })
    resultSet.asScala.foreach(key => jedis.del(key))
    res
  }

}

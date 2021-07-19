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

package com.intel.analytics.zoo.examples.qaranker

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.textmatching.KNRM
import com.intel.analytics.zoo.feature.common.Relations
import com.intel.analytics.zoo.feature.text.TextSet
import scopt.OptionParser


case class QARankerPredictParams(
   dataPath: String = "./", model: String = "",
   wordIndex: String = "", questionLength: Int = 10,
   answerLength: Int = 40, partitionNum: Int = 4,
   batchSize: Int = 200)


object QARankerPredict {
  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[QARankerPredictParams]("QARanker Example") {
      opt[String]("dataPath")
        .required()
        .text("The directory containing the corpus and relations")
        .action((x, c) => c.copy(dataPath = x))
      opt[String]('m', "model")
        .text("The path to the trained KNRM model")
        .action((x, c) => c.copy(model = x))
        .required()
      opt[String]("wordIndex")
        .text("The path to the word dictionary file")
        .action((x, c) => c.copy(wordIndex = x))
        .required()
      opt[Int]("questionLength")
        .text("The sequence length of each question")
        .action((x, c) => c.copy(questionLength = x))
      opt[Int]("answerLength")
        .text("The sequence length of each answer")
        .action((x, c) => c.copy(answerLength = x))
      opt[Int]("partitionNum")
        .text("The number of partitions to cut the dataset into")
        .action((x, c) => c.copy(partitionNum = x))
      opt[Int]('b', "batchSize")
        .text("The number of samples per gradient update")
        .action((x, c) => c.copy(batchSize = x))
    }

    parser.parse(args, QARankerPredictParams()).map { param =>
      val sc = NNContext.initNNContext("QARanker Example")

      val qSet = TextSet.readCSV(param.dataPath + "/question_corpus.csv", sc, param.partitionNum)
        .loadWordIndex(param.wordIndex).tokenize().normalize()
        .word2idx().shapeSequence(param.questionLength)
      val aSet = TextSet.readCSV(param.dataPath + "/answer_corpus.csv", sc, param.partitionNum)
        .loadWordIndex(param.wordIndex).tokenize().normalize()
        .word2idx().shapeSequence(param.answerLength)

      val relations = Relations.read(param.dataPath + "/relation_test.csv",
        sc, param.partitionNum)
      val knrm = KNRM.loadModel(param.model)
      val res = knrm.predictRelationLists(relations, qSet, aSet)
      res.rdd.foreach(x => {
        val prediction = x.getPredict[Float]
        println(prediction)
      })

      sc.stop()
    }
  }
}

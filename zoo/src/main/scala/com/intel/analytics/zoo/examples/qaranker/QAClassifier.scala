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
import com.intel.analytics.zoo.pipeline.api.keras.models.KerasNet
import com.intel.analytics.zoo.pipeline.api.keras.objectives.BinaryCrossEntropy
import com.intel.analytics.zoo.models.textmatching.KNRM
import com.intel.analytics.zoo.feature.common.Relations
import com.intel.analytics.zoo.feature.pmem.MemoryType
import com.intel.analytics.zoo.feature.text.TextSet
import com.intel.analytics.zoo.pipeline.api.keras.metrics.BinaryAccuracy
import com.intel.analytics.zoo.pipeline.api.keras.optimizers.Adam
import scopt.OptionParser


case class QAClassificationParams(
                                   dataPath: String = "./", embeddingFile: String = "./",
                                   questionLength: Int = 10, answerLength: Int = 40,
                                   partitionNum: Int = 4, batchSize: Int = 200,
                                   nbEpoch: Int = 30, learningRate: Double = 0.0005,
                                   model: Option[String] = None, memoryType: String = "DRAM",
                                   outputPath: Option[String] = None)


object QAClassification {
  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[QAClassificationParams]("QAClassification Example") {
      opt[String]("dataPath")
        .required()
        .text("The directory containing the corpus and relations")
        .action((x, c) => c.copy(dataPath = x))
      opt[String]("embeddingFile")
        .required()
        .text("The file path to GloVe embeddings")
        .action((x, c) => c.copy(embeddingFile = x))
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
      opt[Int]('e', "nbEpoch")
        .text("The number of epochs to train the model")
        .action((x, c) => c.copy(nbEpoch = x))
      opt[Double]('l', "learningRate")
        .text("The learning rate for the model")
        .action((x, c) => c.copy(learningRate = x))
      opt[String]('m', "model")
        .text("KNRM model snapshot location if any")
        .action((x, c) => c.copy(model = Some(x)))
      opt[String]("memoryType")
        .text("memory type")
        .action((x, c) => c.copy(memoryType = x))
      opt[String]('o', "outputPath")
        .text("The directory to save the model and word dictionary")
        .action((x, c) => c.copy(outputPath = Some(x)))
    }

    parser.parse(args, QAClassificationParams()).map { param =>
      val sc = NNContext.initNNContext("QAClassification Example")

      val qSet = TextSet.readCSV(param.dataPath + "/question_corpus.csv", sc, param.partitionNum)
        .tokenize().normalize().word2idx(minFreq = 2).shapeSequence(param.questionLength)
      val aSet = TextSet.readCSV(param.dataPath + "/answer_corpus.csv", sc, param.partitionNum)
        .tokenize().normalize().word2idx(minFreq = 2, existingMap = qSet.getWordIndex)
        .shapeSequence(param.answerLength)

      val trainRelations = Relations.read(param.dataPath + "/relation_train.csv",
        sc, param.partitionNum)
      val trainSet = TextSet.fromRelations(trainRelations, qSet, aSet,
        MemoryType.fromString(param.memoryType))
      val validateRelations = Relations.read(param.dataPath + "/relation_valid.csv",
        sc, param.partitionNum)
      val validateSet = TextSet.fromRelations(validateRelations, qSet, aSet,
        MemoryType.fromString(param.memoryType))

      val knrm = if (param.model.isDefined) {
        KNRM.loadModel(param.model.get)
      } else {
        val wordIndex = aSet.getWordIndex
        KNRM(param.questionLength, param.answerLength,
          param.embeddingFile, wordIndex, targetMode = "classification")
      }
      // TODO: refactor this after KNRM extends KerasZooModel
      val model = knrm.model.asInstanceOf[KerasNet[Float]]
      model.compile(optimizer = new Adam[Float](lr = param.learningRate),
        loss = BinaryCrossEntropy[Float](),
        metrics = List(new BinaryAccuracy[Float]()))
      model.fit(trainSet, batchSize = param.batchSize,
        nbEpoch = param.nbEpoch, validationData = validateSet)
      val predictSet = model.predict(validateSet)
      if (param.outputPath.isDefined) {
        val outputPath = param.outputPath.get
        knrm.saveModel(outputPath + "/knrm_classifier.model")
        aSet.saveWordIndex(outputPath + "/word_index.txt")
        println("Trained model and word dictionary saved")
      }
      sc.stop()
    }
  }
}
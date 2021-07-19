#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from zoo.tfpark.text.estimator import *


def _bert_feature_extractor_model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.PREDICT:
        output_layer = bert_model(features, labels, mode, params).get_all_encoder_layers()[-4:]
        # Remark: In the paper, concatenating the last four layers will get the best score.
        # Use add here first due to memory issue.
        features = tf.add_n(output_layer)
        return TFEstimatorSpec(mode=mode, predictions=features)
    else:
        raise ValueError("For feature extraction based on BERT, only PREDICT mode is supported for NER")


class BERTFeatureExtractor(BERTBaseEstimator):
    """
    A pre-built TFEstimator that takes the hidden state of the final encoder layer
    for named entity recognition based on SoftMax classification.

    :param bert_config_file: The path to the json file for BERT configurations.
    :param init_checkpoint: The path to the initial checkpoint of the pre-trained BERT model if any.
                            Default is None.
    :param use_one_hot_embeddings: Boolean. Whether to use one-hot for word embeddings.
                                   Default is False.
    :param optimizer: The optimizer used to train the estimator. It can either be an instance of
                      tf.train.Optimizer or the corresponding string representation.
                      Default is None if no training is involved.
    :param model_dir: The output directory for model checkpoints to be written if any.
                      Default is None.
    """
    def __init__(self, bert_config_file, init_checkpoint=None, use_one_hot_embeddings=False):
        super(BERTFeatureExtractor, self).__init__(
            model_fn=_bert_feature_extractor_model_fn,
            bert_config_file=bert_config_file,
            init_checkpoint=init_checkpoint,
            use_one_hot_embeddings=use_one_hot_embeddings,
            optimizer=None,
            model_dir=None)

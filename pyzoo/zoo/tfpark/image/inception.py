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

from bert import modeling
from nets import inception as slim_inception
import tensorflow as tf

from zoo.tfpark.estimator import TFEstimatorSpec

slim = tf.contrib.slim


def get_inception_v1_model_fn(
        num_classes=1000,
        dropout_keep_prob=0.8,
        prediction_fn=slim.softmax,
        loss_fn=None,
        init_checkpoint=None,
        spatial_squeeze=True,
        global_pool=False):
    def model_fn(features, labels, mode, params):
        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training, keep_prob, has_loss = True, dropout_keep_prob, True
        elif mode == tf.estimator.ModeKeys.EVAL:
            is_training, keep_prob, has_loss = False, 1.0, True
        elif mode == tf.estimator.ModeKeys.PREDICT:
            is_training, keep_prob, has_loss = False, 1.0, False
        else:
            raise ValueError("Invalid mode %s" % mode)

        with slim.arg_scope(slim_inception.inception_v1_arg_scope()):
            logits, end_points = slim_inception.inception_v1(features,
                                                             num_classes=num_classes,
                                                             is_training=is_training,
                                                             dropout_keep_prob=keep_prob,
                                                             prediction_fn=prediction_fn,
                                                             spatial_squeeze=spatial_squeeze,
                                                             global_pool=global_pool)
            if has_loss:
                if loss_fn is None:
                    loss = tf.reduce_mean(
                        tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
                else:
                    loss = loss_fn(logits, labels)
                    loss = tf.reduce_mean(loss)
            else:
                loss = None

        if init_checkpoint:
            tvars = tf.trainable_variables()
            assignment_map, initialized_variable_names = \
                modeling.get_assignment_map_from_checkpoint(tvars, params["init_checkpoint"])
            tf.train.init_from_checkpoint(params["init_checkpoint"], assignment_map)
        return TFEstimatorSpec(mode, predictions=end_points["Predictions"], loss=loss)

    return model_fn


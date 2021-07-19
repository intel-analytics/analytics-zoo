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

import sys

import tensorflow as tf
from nets import lenet

from zoo.tfpark import TFOptimizer

slim = tf.contrib.slim

if __name__ == '__main__':

    export_dir = "/tmp/lenet_export/"
    if len(sys.argv) > 1:
        export_dir = sys.argv[1]

    features = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
    labels = tf.placeholder(dtype=tf.int32, shape=[None])

    with slim.arg_scope(lenet.lenet_arg_scope()):
        logits, end_points = lenet.lenet(features, num_classes=10, is_training=True)

    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))

    # save
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    TFOptimizer.export_training_model(export_dir=export_dir,
                                      loss=loss,
                                      sess=sess,
                                      inputs=(features, labels)
                                      )

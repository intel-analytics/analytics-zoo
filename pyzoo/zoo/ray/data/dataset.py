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

import numpy as np

import tensorflow as tf

class RayDataSet(object):
    def next_batch(self):
        raise Exception("not implemented")

    def get_batchsize(self):
        raise Exception("not implemented")

    @staticmethod
    def from_dataset_generator(input_fn):
        return TFDataSetWrapper(input_fn=input_fn)



class TFDataSetWrapper(RayDataSet):
    def __init__(self, input_fn):
        self.input_fn = input_fn
        self.init = False

    def action(self, force=False):
        if self.init and not force:
            return
        self.init = True
        self.tf_dataset = self.input_fn()
        self.session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=22, inter_op_parallelism_threads=22))
        self.x, self.y = self.tf_dataset.make_one_shot_iterator().get_next()
        return self

    def next_batch(self):
        if not self.init:
            raise Exception("Please invoke init() first")
        return [i for i in self.session.run([self.x, self.y])]


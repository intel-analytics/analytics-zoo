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

import tensorflow as tf
import numpy as np
import ray
import os


from zoo.ray.util.utils import MKLSetting
from zoo.ray.util import utils


@ray.remote(resources={"trainer":1})
class ModelWorker(object):
    def __init__(self, modelLite, ray_data_set, num_workers):
        self.num_workers = num_workers
        self.modelAdapter = modelLite.to_adapter()
        self.ray_data_set = ray_data_set.action()
        self.loss = 0

    # @ray.remote(num_return_vals=2)
    def pull_and_execute(self, *parameters):
        """
        It would return a sharded grads here.
        Each parameter should be a 1-D vector
        """
        flat_parameters = np.concatenate(parameters)
        self.modelAdapter.set_flat_trainable_weights(flat_parameters)

        input_data, label_data = self.ray_data_set.next_batch()

        loss_gradients = self.modelAdapter.execute(utils.to_list(input_data),
                                                   utils.to_list(label_data))
        self.loss = loss_gradients[0]
        grads = loss_gradients[1:]
        print("loss is {}".format(self.loss))
        flat_grads = np.concatenate([g.flatten() for g in grads])
        print("flat_grads {}".format(flat_grads.shape))
        sharded_grads = utils.split(flat_grads, self.num_workers)
        return sharded_grads

    def get_loss(self):
        # TODO: this should be combined with set_parameters_compute_gradients, but we cannot return
        # in this way (loss, List(grad1, grad2))
        return self.loss

    def get_flat_trainable_weights(self):
        return self.modelAdapter.get_flat_weights()


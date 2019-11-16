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

import time

import numpy as np
import ray
import tensorflow as tf
import logging

from zoo.ray.data.dataset import RayDataSet
from zoo.ray.distribute.model import ModelLite
from zoo.ray.distribute.ps import ShardedParameterServer
from zoo.ray.distribute.worker import ModelWorker
from zoo.ray.util import utils

logger = logging.getLogger(__name__)

class RayModel(object):
    """
    You should add your definition at model_fn
    and then return (input, output, target, loss, optimizer)
    """
    def __init__(self, model_bytes=None, model_fn=None):
        self.model_lite = ModelLite(keras_model_bytes= model_bytes,
                                    model_fn = model_fn)

    def from_model_fn(cls, model_fn):
        return cls(model_fn=model_fn)

    @classmethod
    def from_keras_model(cls, keras_model):
        model_bytes = ModelLite.serialize_model(keras_model)
        return cls(model_bytes = model_bytes)

    # TODO: refine val_x, it only accept RayDataset for now
    # how to resume training? reuse some tf existing api?
    def fit(self, x, num_worker, batch_size, y=None, val_x=None, steps=10, strategy="ps"):
        self.batch_size = batch_size
        self.strategy=strategy
        self.num_worker = num_worker
        self.modelAdapter = self.model_lite.to_adapter()
        self.x = self._preprocess_input(x, y)
        self._init_distributed_engine()
        for i in range(1, steps + 1):
            self.step(i)
            if i % 1000 == 0:
                self.modelAdapter.set_flat_trainable_weights(
                    ray.get(self.workers[0].get_flat_weights.remote()))
                self.modelAdapter.save("/opt/work/benchmark/resnet-cifar10-{}".format(i))
                print("acc: {}".format(
                    self.evaluate(x=val_x,
                                      batch_size=10000))) # batch size?
        return self

    def _preprocess_input(self, x, y, batch_size=None, repeat=True, shuffle=True):
            if not batch_size:
                batch_size = int(self.batch_size / self.num_worker)
            #TODO: list of inputs
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                def dataset_fn():
                    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
                    if repeat:
                        dataset = dataset.repeat()
                    if shuffle:
                        dataset = dataset.shuffle(buffer_size= 4 * batch_size)
                    return dataset
                return RayDataSet.from_dataset_generator(
                    input_fn=dataset_fn)
            elif isinstance(x, RayDataSet) and (y is None):
                return x
            else:
                raise TypeError("Unsupported training data type: %s" % type(x))

    def evaluate(self, x, y=None, batch_size=None, metric_fn=None):
        ray_dataset = self._preprocess_input(x, y, batch_size=batch_size, repeat=False)
        # TODO: add metric_fn back
        result = None
        count = 0
        ray_dataset.action(force=True) # we should think of a way to remove force
        try:
            while True:
                input_data, output_data = ray_dataset.next_batch()
                metrics = self.modelAdapter.evaluate(input_data, output_data)
                if not result:
                    result = [0.0] * len(metrics)
                result = [i[0] + i[1] for i in zip(metrics, result)]
                count = count + 1
                print(count)
        except tf.errors.OutOfRangeError:
            pass
        return [r / count for r in result]

    def _init_distributed_engine(self):
        weights = utils.flatten(self.modelAdapter.get_trainable_weights())
        sharded_weights = utils.split(weights, self.num_worker)
        # This weights would be used for both PS and ModelWorker
        sharded_weight_ids = [ray.put(w) for w in sharded_weights]
        self.workers = []
        self.pss = []
        logger.info(
            "Creating parameter server ({} total)".format(
                self.num_worker))

        for ps_index in range(self.num_worker):
            self.pss.append(
                ShardedParameterServer.remote(sharded_weight_ids[ps_index],
                                              modelLite=self.model_lite,
                                              ))

        logger.info(
            "Creating model workers ({} total)".format(
                self.num_worker))
        for worker_index in range(self.num_worker):
            self.workers.append(
                ModelWorker.remote(self.model_lite, self.x, self.num_worker))


    def step(self, step_id):
        start = time.time()
        # workers of sharded_grads
        sharded_grad_ids = []
        results = []
        losses = []
        for worker in self.workers:
            # 1) pull the latest weights from ps
            parameters = [ps.pull.remote() for ps in self.pss]
            # 2) compute the grads
            sharded_grad = worker.pull_and_execute._remote(args=parameters, kwargs=None, num_return_vals=self.num_worker)
            sharded_grad_ids.append(sharded_grad)
            losses.append(worker.get_loss.remote())

        grads_per_ps = list(zip(*sharded_grad_ids))
        assert len(grads_per_ps[0]) == self.num_worker, "we should get correct grads for each ps"
        # 3) push and aggregate grads on ps
        for index, grads in enumerate(grads_per_ps):
            results.append(self.pss[index].push.remote(*grads))
        # wait for complete
        ray.wait(object_ids=results, num_returns=len(results))
        end = time.time()
        avg_loss = np.mean([ray.get(loss) for loss in losses])
        throughput = self.batch_size / (end - start)
        print("Iteration: {}, throughput: {}, loss: {}".format(step_id, throughput, avg_loss))


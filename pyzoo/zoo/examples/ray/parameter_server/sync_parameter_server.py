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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import ray
import model

from zoo import init_spark_on_yarn, init_spark_on_local
from zoo.ray.util.raycontext import RayContext
import os
os.environ["LANG"] = "C.UTF-8"

parser = argparse.ArgumentParser(description="Run the synchronous parameter "
                                             "server example.")
parser.add_argument("--num-workers", default=4, type=int,
                    help="The number of workers to use.")
parser.add_argument("--hadoop_conf", type=str,
                    help="turn on yarn mode by passing the path to the hadoop"
                    "configuration folder. Otherwise, turn on local mode.")

@ray.remote
class ParameterServer(object):
    def __init__(self, learning_rate):
        self.net = model.SimpleCNN(learning_rate=learning_rate)

    def apply_gradients(self, *gradients):
        self.net.apply_gradients(np.mean(gradients, axis=0))
        return self.net.variables.get_flat()

    def get_weights(self):
        return self.net.variables.get_flat()


@ray.remote
class Worker(object):
    def __init__(self, worker_index, batch_size=50):
        self.worker_index = worker_index
        self.batch_size = batch_size
        self.mnist = model.download_mnist_retry(seed=worker_index)
        self.net = model.SimpleCNN()

    def compute_gradients(self, weights):
        self.net.variables.set_flat(weights)
        xs, ys = self.mnist.train.next_batch(self.batch_size)
        return self.net.compute_gradients(xs, ys)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.hadoop_conf:
        slave_num = 2
        sc = init_spark_on_yarn(
            hadoop_conf=args.hadoop_conf,
            conda_name="ray36",
            num_executor=slave_num,
            executor_cores=28,
            executor_memory="10g",
            driver_memory="2g",
            driver_cores=4,
            extra_executor_memory_for_ray="30g")
        ray_ctx = RayContext(sc=sc, object_store_memory="25g")
    else:
        sc = init_spark_on_local(cores=4)
        ray_ctx = RayContext(sc=sc)

    ray_ctx.init()

    # Create a parameter server.
    net = model.SimpleCNN()
    ps = ParameterServer.remote(1e-4 * args.num_workers)

    # Create workers.
    workers = [Worker.remote(worker_index)
               for worker_index in range(args.num_workers)]

    # Download MNIST.
    mnist = model.download_mnist_retry()

    i = 0
    current_weights = ps.get_weights.remote()
    while True:
        # Compute and apply gradients.
        gradients = [worker.compute_gradients.remote(current_weights)
                     for worker in workers]
        current_weights = ps.apply_gradients.remote(*gradients)

        if i % 10 == 0:
            # Evaluate the current model.
            net.variables.set_flat(ray.get(current_weights))
            test_xs, test_ys = mnist.test.next_batch(1000)
            accuracy = net.compute_accuracy(test_xs, test_ys)
            print("Iteration {}: accuracy is {}".format(i, accuracy))
        i += 1
    ray_ctx.stop()

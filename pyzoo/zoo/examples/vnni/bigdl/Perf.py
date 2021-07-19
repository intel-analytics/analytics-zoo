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
from zoo.models.image.imageclassification import ImageClassifier
from bigdl.util.common import init_engine
from optparse import OptionParser
import numpy as np
import time


def perf(model_path, batch_size, iteration):
    batch_input = np.random.rand(batch_size, 3, 224, 224)
    single_input = np.random.rand(1, 3, 224, 224)
    init_engine()

    model = ImageClassifier.load_model(model_path)
    model.set_evaluate_status()

    for i in range(iteration):
        start = time.time_ns()
        model.forward(batch_input)
        time_used = time.time_ns() - start
        throughput = round(batch_size / (time_used / 10 ** 6), 2)
        print("Iteration:" + str(i) +
              ", batch " + str(batch_size) +
              ", takes " + str(time_used) + " ms" +
              ", throughput is " + str(throughput) + " imgs/sec")

    # mkldnn model would forward a fixed batch size.
    # Thus need a new model to test for latency.
    model2 = ImageClassifier.load_model(model_path)
    model2.set_evaluate_status()

    for i in range(iteration):
        start = time.time_ns()
        model.forward(single_input)
        latency = time.time_ns() - start
        print("Iteration:" + str(i) +
              ", latency for a single image is " + str(latency / 10 ** 6) + " ms")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--model", type=str, dest="model_path",
                      help="The path to the downloaded int8 model snapshot")
    parser.add_option("--batch_size", type=int, dest="batch_size", default=32,
                      help="The batch size of input data")
    parser.add_option("--iteration", type=int, dest="iteration", default=100,
                      help="The number of iterations to run the performance test. "
                           "The result should be the average of each iteration time cost")

    (options, args) = parser.parse_args(sys.argv)
    perf(options.model_path, options.batch_size, options.iteration)

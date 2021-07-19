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

from zoo.serving.client import InputQueue, OutputQueue
import time

def run(path):
    input_api = InputQueue()
    base_path = path

    if not base_path:
        raise EOFError("You have to set your image path")
    output_api = OutputQueue()
    output_api.dequeue()

    import numpy as np
    a = np.array([2])
    b = np.array([10])
    c = np.array([1])
    d = np.array([1])
    e = np.array([1])
    input_api.enqueue('tftest', t1=a, t2=b,t3=c,t4=d,t5=e)

    time.sleep(10)
    print(output_api.query('tftest'))

if __name__ == "__main__":
    run("nothing")

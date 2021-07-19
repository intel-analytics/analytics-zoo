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
    a = np.array([[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0]])
    b = np.array([[1.0,745.0,0.0,10.0]])
    sparse_tensor = [np.array([[0, 0, 0,0,0,0,0,0], [1,4,21,293,532,815,1237,1693]]),
                     np.array([1,1,1,1,1,1,1,1]),
                     np.array([1, 2051])]
    d = np.array([[1.0]])

    input_api.enqueue('wndtest',t0=sparse_tensor, t1=a,t2=d,t3=b)

    time.sleep(10)
    print(output_api.query('wndtest'))

if __name__ == "__main__":
    run("nothing")

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
import numpy as np
# input tensor
x = np.array([2, 3], dtype=np.float32)

input_api = InputQueue()
input_api.enqueue('my_input', input=x)

output_api = OutputQueue()
result = output_api.query('my_input')
print("Result is :" + str(result))

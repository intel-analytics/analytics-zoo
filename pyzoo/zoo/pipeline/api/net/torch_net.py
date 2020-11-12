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
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from pyspark import RDD

from bigdl.nn.layer import Layer
from zoo import getOrCreateSparkContext
from zoo.common.utils import callZooFunc
from zoo.feature.image import ImageSet

if sys.version >= '3':
    long = int
    unicode = str


class TorchNet(Layer):
    """
    TorchNet wraps a TorchScript model as a single layer, thus the Pytorch model can be used for
    distributed inference or training.
    :param path: path to the TorchScript model.
    """

    def __init__(self, path, bigdl_type="float"):
        super(TorchNet, self).__init__(None, bigdl_type, path)

    @staticmethod
    def from_pytorch(module, input, dtype=torch.float, check_trace=True):
        """
        Create a TorchNet directly from PyTorch model, e.g. model in torchvision.models.
        Users need to provide an example input or the input tensor shape.
        :param module: a PyTorch model
        :param input: To trace the tensor operations, torch jit trace requires users to
                      provide an example input. Here the input parameter can be:
                        1. a torch tensor, or tuple of torch tensors for multi-input models
                        2. list of integers, or tuple of int list for multi-input models. E.g. For
                           ResNet, this can be [1, 3, 224, 224]. A random tensor with the
                           specified size will be used as the example input.
        :param check_trace: boolean value, optional. check if the same inputs run through
                            traced module produce the same outputs. Default: ``True``. You
                            might want to disable this if, for example, your network contains
                            non-deterministic ops or if you are sure that the network is
                            correct despite a checker failure.
        """
        if input is None:
            raise Exception("please provide an example input or input Tensor size")

        sample = TorchNet.get_sample_input(input, dtype)
        temp = tempfile.mkdtemp()

        from torch import nn

        class OuterModule(nn.Module):
            def __init__(self, model):
                super(OuterModule, self).__init__()
                self.module = model

            def forward(self, x):
                if dtype == torch.long:
                    x = x.long()
                x = self.module(x)
                return x

        # save model
        traced_script_module = torch.jit.trace(OuterModule(module), sample, check_trace=check_trace)
        path = os.path.join(temp, "model.pt")
        traced_script_module.save(path)

        net = TorchNet(path)
        shutil.rmtree(temp)

        return net

    @staticmethod
    def get_sample_input(input, dtype):
        if isinstance(input, torch.Tensor):
            return input

        elif isinstance(input, (list, tuple)) and len(input) > 0:
            if all(isinstance(x, torch.Tensor) for x in input):  # tensors
                return tuple(input)
            elif all(isinstance(x, int) for x in input):  # ints
                return generate_torch_input(input, dtype)
            elif all(isinstance(x, (list, tuple)) for x in input) and \
                    all(isinstance(y, int) for x in input for y in x):  # nested int list (tuple)
                return tuple(map(lambda size: generate_torch_input(size, dtype), input))

        raise Exception("Unsupported input type: " + str(input))

    def save_pytorch(self, path):
        '''
        save the model as a torch script module
        '''
        pythonBigDL_method_name = "torchNetSavePytorch"
        callZooFunc(self.bigdl_type, pythonBigDL_method_name, self.value, path)
        return

    def predict(self, x, batch_per_thread=1, distributed=True):
        """
        Use a model to do prediction.
        """
        if isinstance(x, ImageSet):
            results = callZooFunc(self.bigdl_type, "zooPredict",
                                  self.value,
                                  x,
                                  batch_per_thread)
            return ImageSet(results)
        if distributed:
            if isinstance(x, np.ndarray):
                from zoo.util.utils import to_sample_rdd
                data_rdd = to_sample_rdd(x, np.zeros([x.shape[0]]), getOrCreateSparkContext())
            elif isinstance(x, RDD):
                data_rdd = x
            else:
                raise TypeError("Unsupported prediction data type: %s" % type(x))
            results = callZooFunc(self.bigdl_type, "zooPredict",
                                  self.value,
                                  data_rdd,
                                  batch_per_thread)
            return results.map(lambda result: Layer.convert_output(result))
        else:
            if isinstance(x, np.ndarray) or isinstance(x, list):
                results = callZooFunc(self.bigdl_type, "zooPredict",
                                      self.value,
                                      self._to_jtensors(x),
                                      batch_per_thread)
                return [Layer.convert_output(result) for result in results]
            else:
                raise TypeError("Unsupported prediction data type: %s" % type(x))


def generate_torch_input(size, dtype):
    if dtype == torch.long:
        return torch.LongTensor(size).random_(0, 1)
    else:
        return torch.rand(size)

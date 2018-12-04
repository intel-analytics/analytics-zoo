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


from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.utils import remove_batch
from bigdl.util.common import to_list, callBigDlFunc

if sys.version >= '3':
    long = int
    unicode = str


class Sequential(Sequential):
    """
    Container for a sequential model.

    # Arguments
    name: String to specify the name of the sequential model. Default is None.

    >>> sequential = Sequential(name="seq1")
    creating: createZooKerasSequential
    """
    def __init__(self, jvalue=None, **kwargs):
        super(Sequential, self).__init__(jvalue, **kwargs)

    # TODO: expose is_built from scala side
    def is_built(self):
        try:
            self.get_output_shape()
            return True
        except:
            return False

    def add(self, model):
        from zoo.pipeline.api.autograd import Lambda
        if (isinstance(model, Lambda)):
            if not self.is_built():
                if not model.input_shape:
                    raise Exception("You should specify inputShape for the first layer")
                input_shapes = model.input_shape
            else:
                input_shapes = self.get_output_shape()
            model = model.create(remove_batch(input_shapes))
        self.value.add(model.value)
        return self

    @staticmethod
    def from_jvalue(jvalue, bigdl_type="float"):
        """
        Create a Python Model base on the given java value
        :param jvalue: Java object create by Py4j
        :return: A Python Model
        """
        model = Sequential(jvalue=jvalue)
        model.value = jvalue
        return model


class Model(Model):
    """
    Container for a graph model.

    # Arguments
    inputs: An input node or a list of input nodes.
    outputs: An output node or a list of output nodes.
    name: String to specify the name of the graph model. Default is None.
    """
    def __init__(self, inputs, outputs, jvalue=None, **kwargs):
        super(Model, self).__init__(jvalue,
                                    to_list(inputs),
                                    to_list(outputs),
                                    **kwargs)

    def save_graph_topology(self, log_path, backward=False):
        """
        Save the current model graph to a folder, which can be displayed in TensorBoard
        by running the command:
        tensorboard --logdir log_path

        # Arguments
        log_path: The path to save the model graph.
        backward: The name of the application.
        """
        callBigDlFunc(self.bigdl_type, "zooSaveGraphTopology",
                      self.value,
                      log_path,
                      backward)

    def new_graph(self, output):
        value = callBigDlFunc(self.bigdl_type, "newGraph", self.value, output)
        return self.from_jvalue(value)

    def freeze_up_to(self, names):
        callBigDlFunc(self.bigdl_type, "freezeUpTo", self.value, names)

    def unfreeze(self, names):
        callBigDlFunc(self.bigdl_type, "unFreeze", self.value, names)

    @staticmethod
    def from_jvalue(jvalue, bigdl_type="float"):
        """
        Create a Python Model base on the given java value
        :param jvalue: Java object create by Py4j
        :return: A Python Model
        """
        model = Model([], [], jvalue=jvalue)
        model.value = jvalue
        return model

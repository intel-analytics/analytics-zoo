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

import logging
import os
import tempfile
import time

import numpy as np
import ray
import tensorflow as tf

from zoo.ray.data.dataset import RayDataSet
from zoo.ray.distribute.ps import ShardedParameterServer
from zoo.ray.distribute.tfvariableupdater import TFVariableUpdater
from zoo.ray.distribute.worker import ModelWorker
from zoo.ray.util import utils
from zoo.ray.util.utils import MKLSetting, unflatten
import tensorflow.keras.backend as K

logger = logging.getLogger(__name__)

class IModel():
    def execute(self,features, labels):
        """

        :param features:
        :param labels:
        :return: loss and gradients
        """
        raise Exception("Not implement yet")

    def set_flat_trainable_weights(self, flat_weights):
        """

        :param flat_weights: 1D tensor
        :return:
        """
        raise Exception("Not implement yet")

    def get_trainable_weights(self):
        """
        :return: list of ndarray
        """
        raise Exception("Not implement yet")

    def apply_grads(self, grads):
        """
        :param grads: list of ndarray
        :return:
        """
        raise Exception("Not implement yet")

class KerasModelImpl(IModel, MKLSetting):

    def __init__(self, kerasModel, mklCores=None):
        super().__init__(cores = mklCores)
        self.kerasModel = kerasModel

        try:
            self.loss = kerasModel.total_loss
            self.inputs = utils.to_list(kerasModel.inputs)
            self.targets = utils.to_list(kerasModel._targets)
            self.trainable_vars = kerasModel.trainable_weights
            self.non_trainable_vars = kerasModel.non_trainable_weights

            self.grads = K.gradients(self.loss, self.trainable_vars)
            self.optimizer = kerasModel.optimizer
            self.sess = K.get_session()
            self.weightShapes = [v.get_shape().as_list() for v in self.trainable_vars]

            ww = self.kerasModel.get_weights()
            self.tfVariableUpdater = TFVariableUpdater(sess=self.sess, vars=self.trainable_vars + self.non_trainable_vars)
            flat_ww = self.tfVariableUpdater.get_flat()
            # self.tfVariableUpdater.set_flat()

            self.sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=self.get_cores(),
                    inter_op_parallelism_threads=self.get_cores()))
            K.set_session(self.sess)
            self.sess.run(tf.global_variables_initializer())
            # self.kerasModel.set_weights(ww)
            self.tfVariableUpdater = TFVariableUpdater(sess=self.sess, vars=self.trainable_vars + self.non_trainable_vars)
            self.tfVariableUpdater.set_flat(flat_ww)  # still testing this behavior
            flat_ww2 = self.get_flat_weights()
            flat_ww2

            self.tfVariableUpdater = TFVariableUpdater(sess=self.sess, vars=self.trainable_vars)

        except Exception as e:
            # print stack here.
            raise e


    def _generate_feed_dict(self, inputs_op, inputs, targets_op, targets):
        fdict = {}
        if inputs:
            fdict.update(dict(zip(inputs_op, inputs)))
        if targets:
            fdict.update(dict(zip(targets_op, targets)))
        return fdict

    def execute(self, features, labels):
        """
        :param features:
        :param labels:
        :return: loss and gradients
        """
        loss_gradients = self._eval(features, labels, tensors=[self.loss] + self.grads)
        return loss_gradients

    def _eval(self, features, labels, tensors):
        feed_dict_data = self._generate_feed_dict(self.inputs,
                                                  utils.to_list(features),
                                                  self.targets,
                                                  utils.to_list(labels))
        tensors_value = self.sess.run(
            utils.to_list(tensors),
            feed_dict=feed_dict_data)
        return tensors_value


    def set_flat_trainable_weights(self, flat_weights):
        """

        :param flat_weights: 1D tensor
        :return:
        """

        # arrays = unflatten(flat_weights, self.weightShapes)
        self.tfVariableUpdater.set_flat(flat_weights)

    def get_flat_weights(self):
        return self.tfVariableUpdater.get_flat()

    def get_trainable_weights(self):
        """
        :return: list of ndarray
        """
        return K.batch_get_value(self.trainable_vars)

    def evaluate(self, inputs, targets):
        metrics_tensors = [
            self.kerasModel._all_metrics_tensors[m] for m in self.kerasModel.metrics_names[1:]
        ]
        return self._eval(inputs, targets, metrics_tensors)

    def save(self, model_path):
        self.kerasModel.save(model_path)




class ModelAdapter(MKLSetting):

    def __init__(self, inputs,
                 outputs,
                 targets,
                 loss,
                 optimizer,
                 grad_vars,
                 cores=None):
        super().__init__(cores)
        self.inputs = inputs
        self.outputs = outputs
        self.targets = targets
        self.loss = loss
        self.optimizer = optimizer
        self.sess = tf.keras.backend.get_session()

        self.sess.run(tf.global_variables_initializer())

    def calc_accuracy(sess, inputs_op, outputs_op, targets_op, input_data, output_data):
        with tf.name_scope('accuracy'):
            # label [-1, 1] not one-hot encoding. If the shape mismatch, the result would be incorrect
            # as `tf.equal` would broadcast automatically during the comparing stage.
            correct_prediction = tf.equal(tf.argmax(outputs_op[0], 1),
                                          tf.cast(tf.reshape(targets_op[0], (-1,)), tf.int64))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)
            return sess.run(accuracy,
                            feed_dict={targets_op[0]: output_data, inputs_op[0]: input_data})


class ModelLite(object):
    def __init__(self, keras_model_bytes=None, model_fn=None):
        self.keras_model_bytes = keras_model_bytes
        self.model_fn = model_fn

    def to_adapter(self):
        if self.model_fn:
            return self.extract_from_model_fn()
        elif self.keras_model_bytes:
            kerasModel = ModelLite.deserialize_model(self.keras_model_bytes)
            return KerasModelImpl(kerasModel)
        else:
            raise Exception("A corrupted ModelLite")

    def extract_from_model_fn(self):
        input, output, target, loss, optimizer = self.model_fn()
        # A list of (gradient, variable) pairs
        grad_vars = [
            t for t in optimizer.compute_gradients(loss)
            if t[0] is not None
        ]
        return input, output, target, loss, optimizer, grad_vars

    @staticmethod
    def deserialize_model(model_bytes):
        model_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_dir, "model.h5")
        try:
            with open(model_path, "wb") as f:
                f.write(model_bytes)
        # TODO: remove file and add more exception handling
        except Exception as e:
            raise e
        loaded_model = tf.keras.models.load_model(model_path)
        return loaded_model

        import pickle
        return pickle.loads(model_bytes)

    @staticmethod
    def serialize_model(model):
        """Serialize model into byte array."""
        try:
            model_dir = tempfile.mkdtemp()
            model_path = os.path.join(model_dir, "model.h5")
            # TODO: change to definition only.
            model.save(str(model_path))
            with open(model_path, "rb") as file:
                return file.read()
        finally:
            # TODO: remove file here
            pass



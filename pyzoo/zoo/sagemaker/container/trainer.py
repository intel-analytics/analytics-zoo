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

import boto3
import inspect
import os
import tensorflow as tf
# from tf_container.run import logger
# import tf_container.s3_fs as s3_fs
import logging

logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    level=logging.INFO)

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Trainer(object):
    DEFAULT_TRAINING_CHANNEL = 'training'

    def __init__(self,
                 customer_script,
                 input_channels=None,
                 model_path=None,
                 output_path=None,
                 customer_params={},
                 checkpoint_path=None):
        """

        Args:
            customer_script: (module) Customer loaded module
            input_channels: (dict) Dictionary with input channels
            model_path: (str) Directory where model will be saved. Can be a S3 bucket
            output_path: (str) Local directory where the output will be saved
            checkpoint_path: (str) Directory where checkpoints will be saved. Can be a S3 bucket
        """
        self.customer_script = customer_script
        self.input_channels = input_channels
        self.model_path = model_path
        self.ouput_path = output_path
        self.customer_params = customer_params
        self.checkpoint_path = checkpoint_path

    def train(self):
        self.train_model()

    def train_model(self):
        hyperparameters = self.customer_params

        if hasattr(self.customer_script, 'train_fn'):
            logger.info('invoking the user-provided train_fn')
            return self.customer_script.train_fn(self.input_channels, hyperparameters, self.model_path, self.checkpoint_path)

    def _build_train_spec(self):
        invoke_args = self._resolve_input_fn_args(self.customer_script.train_input_fn)
        train_input_fn = lambda: self.customer_script.train_input_fn(**invoke_args)
        return train_input_fn

    def _resolve_input_fn_args(self, customer_fn):
        declared_args = inspect.getargspec(customer_fn)
        return {arg: self._resolve_input_fn_param_value(arg) for arg in declared_args.args}

    def _resolve_input_fn_param_value(self, alias_key):
        """
        Handle potentially aliased key name and return value for valid one.

        :return: value for the requested parameter or None
        """
        key_mappings = {('training_dir', 'dir'): 'training_dir',
                        ('hyperparameters', 'params'): 'hyperparameters',
                        ('input_channels', 'channels'): 'input_channels'}
        resolved_key = None
        for k, v in key_mappings.items():
            if alias_key in k:
                resolved_key = v
                break

        parameter_values = {'training_dir': self.input_channels.get(self.DEFAULT_TRAINING_CHANNEL, None),
                            'hyperparameters': self.customer_params,
                            'input_channels': self.input_channels}
        return parameter_values[resolved_key] if resolved_key else None
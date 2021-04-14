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

from zoo.zouwu.model.forecast.abstract import Forecaster
from zoo.zouwu.model.forecast.utils import *
from zoo.zouwu.model.tcn import TCNPytorch

from zoo.orca import init_orca_context
from zoo.orca.learn.pytorch import Estimator
from zoo.orca.learn.metrics import *


class TCNForecaster(Forecaster):

    def __init__(self,
                 past_seq_len,
                 future_seq_len,
                 input_feature_num,
                 output_feature_num,
                 num_channels=[30]*8,
                 kernel_size=7,
                 dropout=0.2,
                 optimizer="Adam",
                 loss="mse",
                 lr=0.001):
        """
        Build a TCN Forecast Model.

        :param past_seq_len: Specify the history time steps (i.e. lookback).
        :param future_seq_len: Specify the output time steps (i.e. horizon).
        :param input_feature_num: Specify the feature dimension.
        :param output_feature_num: Specify the output dimension.
        :param num_channels: Specify the convolutional layer filter number in
               TCN's encoder. This value defaults to [30]*8.
        :param kernel_size: Specify convolutional layer filter height in TCN's
               encoder. This value defaults to 7.
        :param dropout: Specify the dropout close possibility (i.e. the close
               possibility to a neuron). This value defaults to 0.2.
        :param optimizer: Specify the optimizer used for training. This value
               defaults to "Adam".
        :param loss: Specify the loss function used for training. This value
               defaults to "mse". You can choose from "mse", "mae" and
               "huber_loss".
        :param lr: Specify the learning rate. This value defaults to 0.001.
        """
        self.internal = TCNPytorch(check_optional_config=False)
        self.data_config = {
            "past_seq_len": past_seq_len,
            "future_seq_len": future_seq_len,
            "input_feature_num": input_feature_num,
            "output_feature_num": output_feature_num
        }
        self.config = {
            "lr": lr,
            "loss": loss,
            "num_channels": num_channels,
            "kernel_size": kernel_size,
            "optim": optimizer,
            "dropout": dropout
        }

    def fit(self, data, validation_data=None, epochs=1, metric="mse", batch_size=32,
            distributed=False, **kwargs):
        """
        Fit(Train) the forecaster.

        :param data: For single node training, it's a dictionary.
        x: A numpy array with shape (num_samples, lookback, feature_dim).
        lookback and feature_dim should be the same as past_seq_len and input_feature_num.
        y: A numpy array with shape (num_samples, horizon, target_dim).
        horizon and target_dim should be the same as future_seq_len and output_feature_num.
        
        For distributed training, it can be an instance of SparkXShards, a Spark DataFrame
        or a function that takes config and batch_size as argument and returns a PyTorch
        DataLoader for training
        :param validation_data: validation data, similar to data
        :param epochs: Number of epochs you want to train.
        :param metric: The metric for training data.
        :param batch_size: Number of batch size you want to train.
        :param distributed: whether train in ditributed.
        """
        self.config["batch_size"] = batch_size

        if not distributed:
            x = data["x"]
            y = data["y"]

            if validation_data:
                val_x = validation_data["x"]
                val_y = validation_data["y"]
                v_data = (val_x, val_y)
            else:
                v_data = (x, y)
            self._check_data(x, y)
            res = self.internal.fit_eval(x,
                                         y,
                                         validation_data=v_data,
                                         epochs=epochs,
                                         metric=metric,
                                         **self.config)
        else:
            cluster_mode = kwargs.get("cluster_mode", "local")
            num_nodes = kwargs.get("num_nodes", 1)
            cores = kwargs.get("cores", 2)
            memory = kwargs.get("memory", "2g")
            init_orca_context(
                cluster_mode=cluster_mode, num_nodes=num_nodes, cores=cores, memory=memory, **kwargs)

            self.config.update(self.data_config)
            self.internal.build(self.config)

            orca_metrics = convert_str_metric_to_orca_metric(metric)
            self.orca_estimator = Estimator.from_torch(model=self.internal.model,
                                                  optimizer=self.internal.optimizer,
                                                  loss=self.internal.criterion,
                                                  backend="bigdl",
                                                  metrics=orca_metrics)
            res = self.orca_estimator.fit(data=data, epochs=epochs, validation_data=validation_data)

        return res

    def _check_data(self, x, y):
        assert self.data_config["past_seq_len"] == x.shape[-2], \
            "The x shape should be (batch_size, past_seq_len, input_feature_num), \
            Got past_seq_len of {} in config while x input shape of {}."\
            .format(self.data_config["past_seq_len"], x.shape[-2])
        assert self.data_config["future_seq_len"] == y.shape[-2], \
            "The y shape should be (batch_size, future_seq_len, output_feature_num), \
            Got future_seq_len of {} in config while y input shape of {}."\
            .format(self.data_config["future_seq_len"], y.shape[-2])
        assert self.data_config["input_feature_num"] == x.shape[-1],\
            "The x shape should be (batch_size, past_seq_len, input_feature_num), \
            Got input_feature_num of {} in config while x input shape of {}."\
            .format(self.data_config["input_feature_num"], x.shape[-1])
        assert self.data_config["output_feature_num"] == y.shape[-1], \
            "The y shape should be (batch_size, future_seq_len, output_feature_num), \
            Got output_feature_num of {} in config while y input shape of {}."\
            .format(self.data_config["output_feature_num"], y.shape[-1])

    def predict(self, x, distributed=False):
        """
        Predict using a trained forecaster.

        :param x: for single mode: A numpy array with shape (num_samples, lookback, feature_dim).
                  for distributed mode: An instance of SparkXShards or a Spark DataFrame
        """
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        if not distributed:
            return self.internal.predict(x)
        else:
            assert self.orca_estimator
            return self.orca_estimator.predict(x)

    def predict_with_onnx(self, x, dirname=None):
        """
        Predict using a trained forecaster with onnxruntime.

        :param x: A numpy array with shape (num_samples, lookback, feature_dim).
        :param dirname: The directory to save onnx model file. This value defaults
               to None for no saving file.
        """
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        return self.internal.predict_with_onnx(x, dirname=dirname)

    def evaluate(self, val_data, metrics=['mse'], distributed=False):
        """
        Evaluate using a trained forecaster.

        :param val_data: For single node training, it's a dictionary.
        x: A numpy array with shape (num_samples, lookback, feature_dim).
        y: A numpy array with shape (num_samples, horizon, target_dim).
        
        For distributed training, it can be an instance of SparkXShards, a Spark DataFrame
        or PyTorch DataLoader and PyTorch DataLoader creator function
        :param metrics: A list contains metrics for test/valid data.
        """
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling evaluate!")
        if not distributed:
            x = val_data["x"]
            y = val_data["y"]
            return self.internal.evaluate(x, y, metrics=metrics)
        else:
            orca_metrics = convert_str_metric_to_orca_metric(metrics)
            self.orca_estimator.metrics = Metric.convert_metrics_list(orca_metrics)
            return self.orca_estimator.evaluate(val_data)

    def evaluate_with_onnx(self, x, y, metrics=['mse'], dirname=None):
        """
        Evaluate using a trained forecaster with onnxruntime.

        :param x: A numpy array with shape (num_samples, lookback, feature_dim).
        :param y: A numpy array with shape (num_samples, horizon, target_dim).
        :param metrics: A list contains metrics for test/valid data.
        :param dirname: The directory to save onnx model file. This value defaults
               to None for no saving file.
        """
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling evaluate!")
        return self.internal.evaluate_with_onnx(x, y, metrics=metrics, dirname=dirname)

    def save(self, checkpoint_file):
        """
        Save the forecaster.

        :param checkpoint_file: The location you want to save the forecaster.
        """
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling save!")
        self.internal.save(checkpoint_file)

    def restore(self, checkpoint_file):
        """
        restore the forecaster.

        :param checkpoint_file: The checkpoint file location you want to load the forecaster.
        """
        self.internal.restore(checkpoint_file)

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

import numpy as np
import tempfile
import os

from zoo.zouwu.model.forecast.tcn_forecaster import TCNForecaster
from unittest import TestCase
import pytest

num_train_samples = 1000
num_val_samples = 400
num_test_samples = 400
input_feature_dim = 1
output_feature_dim = 1


def create_data():
    input_time_steps = 24
    output_time_steps = 5

    def get_x_y(num_samples):
        x = np.random.rand(num_samples, input_time_steps, input_feature_dim)
        y = x[:, -output_time_steps:, :]*2 + \
            np.random.rand(num_samples, output_time_steps, output_feature_dim)
        return {"x":x, "y":y}

    train_data = get_x_y(num_train_samples)
    val_data = get_x_y(num_val_samples)
    test_data = get_x_y(num_test_samples)
    return train_data, val_data, test_data


def create_dataloader():
    import torch
    from torch.utils.data import TensorDataset
    input_time_steps = 5
    output_time_steps = 2
    inputs = torch.rand((num_train_samples, input_time_steps, input_feature_dim))
    targets = torch.rand(num_train_samples, output_time_steps, output_feature_dim)
    train_loader = torch.utils.data.DataLoader(
        TensorDataset(inputs, targets),
        batch_size=2,
    )
    val_loader = torch.utils.data.DataLoader(
        TensorDataset(inputs, targets),
        batch_size=2,
    )
    return train_loader, val_loader


class TestZouwuModelTCNForecaster(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_tcn_forecaster_fit_eva_pred(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   loss="mae",
                                   lr=0.01)
        train_loss = forecaster.fit(train_data, epochs=2)
        test_pred = forecaster.predict(test_data["x"])
        assert test_pred.shape == test_data["y"].shape
        test_mse = forecaster.evaluate(test_data)

    def test_tcn_forecaster_fit_eva_distribute(self):
        train_data, val_data = create_dataloader()
        forecaster = TCNForecaster(past_seq_len=5,
                                   future_seq_len=2,
                                   input_feature_num=input_feature_dim,
                                   output_feature_num=output_feature_dim,
                                   kernel_size=3,
                                   num_channels=[4, 4],
                                   loss="mae",
                                   lr=0.01)
        train_loss = forecaster.fit(data=train_data, epochs=1,
                                   distributed=True)
        test_mse = forecaster.evaluate(val_data=val_data, metrics="mse",
                                   distributed=True)

    def test_tcn_forecaster_onnx_methods(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   lr=0.01)
        forecaster.fit(train_data, epochs=2)
        try:
            import onnx
            import onnxruntime
            pred = forecaster.predict(test_data["x"])
            pred_onnx = forecaster.predict_with_onnx(test_data["x"])
            np.testing.assert_almost_equal(pred, pred_onnx, decimal=5)
            mse = forecaster.evaluate(test_data)
            mse_onnx = forecaster.evaluate_with_onnx(test_data)
            np.testing.assert_almost_equal(mse, mse_onnx, decimal=5)
        except ImportError:
            pass

    def test_tcn_forecaster_save_restore(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=4,
                                   num_channels=[16, 16],
                                   lr=0.01)
        train_mse = forecaster.fit(train_data, epochs=2)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "ckpt")
            test_pred_save = forecaster.predict(test_data["x"])
            forecaster.save(ckpt_name)
            forecaster.restore(ckpt_name)
            test_pred_restore = forecaster.predict(test_data["x"])
        np.testing.assert_almost_equal(test_pred_save, test_pred_restore)

    def test_tcn_forecaster_runtime_error(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=1,
                                   kernel_size=3,
                                   lr=0.01)
        with pytest.raises(RuntimeError):
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                ckpt_name = os.path.join(tmp_dir_name, "ckpt")
                forecaster.save(ckpt_name)
        with pytest.raises(RuntimeError):
            forecaster.predict(test_data["x"])
        with pytest.raises(RuntimeError):
            forecaster.evaluate(test_data["x"], test_data["x"])

    def test_tcn_forecaster_shape_error(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(past_seq_len=24,
                                   future_seq_len=5,
                                   input_feature_num=1,
                                   output_feature_num=2,
                                   kernel_size=3,
                                   lr=0.01)
        with pytest.raises(AssertionError):
            forecaster.fit(train_data, epochs=2)

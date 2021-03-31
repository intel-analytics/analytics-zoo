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
from zoo.zouwu.model.forecast.model.abstract import BaseModel
from zoo.automl.common.util import *
from zoo.automl.common.metrics import Evaluator
import pickle
import tensorflow as tf
from tensorflow.keras import backend as K


def check_tf_version():
    tf_version = tf.__version__
    if tf_version >= "2.0":
        raise RuntimeError(f"Currently running TensorFlow version {tf_version}. We only support"
                           f"TensorFlow 1.x for now and has been tested on 1.15")


class KerasBaseModel(BaseModel):
    def __init__(self,
                 model_creator,
                 check_optional_config=False):
        self.check_optional_config = check_optional_config
        self.model_creator = model_creator
        self.model = None
        self.config = None
        self.model_built = False

    def build(self, config):
        # update config
        self._check_config(**config)
        self.config = config
        # build model
        self.model = self.model_creator(config)
        self.model_built = True

    def fit_eval(self, x, y, validation_data=None, mc=False, verbose=0, epochs=1, metric="mse",
                 **config):
        """
        fit_eval will build a model at the first time it is built
        config will be updated for the second or later times with only non-model-arch
        params be functional
        TODO: check the updated params and decide if the model is needed to be rebuilt
        """
        def update_config():
            # LSTM & MTNet
            config.setdefault("input_dim", x.shape[-1])
            config.setdefault("output_dim", y.shape[-1])
            # Seq2Seq
            config.setdefault("future_seq_len", y.shape[1])
            config.setdefault("output_feature_num", y.shape[2])
            config.setdefault("input_feature_num", x.shape[2])
            # other
            config.update({"metric": metric})

        if not self.model_built:
            update_config()
            self.build(config)
        else:
            tmp_config = self.config.copy()
            tmp_config.update(config)
            self._check_config(**tmp_config)
            self.config.update(config)

        hist = self.model.fit(x, y,
                              validation_data=validation_data,
                              batch_size=self.config.get("batch_size", 32),
                              epochs=epochs,
                              verbose=verbose
                              )

        hist_metric_name = tf.keras.metrics.get(metric).__name__
        if validation_data is None:
            result = hist.history.get(hist_metric_name)[-1]
        else:
            result = hist.history.get('val_' + hist_metric_name)[-1]
        return result

    def evaluate(self, x, y, metrics=['mse']):
        """
        Evaluate on x, y
        :param x: input
        :param y: target
        :param metrics: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        y_pred = self.predict(x)
        return [Evaluator.evaluate(m, y, y_pred) for m in metrics]

    def predict(self, x):
        """
        Prediction on x.
        :param x: input
        :return: predicted y
        """
        if not self.model_built:
            raise RuntimeError("You must call fit_eval or restore first before calling predict!")
        return self.model.predict(x, batch_size=self.config.get("batch_size", 32))

    def predict_with_uncertainty(self, x, n_iter=100):
        if not self.model_built:
            raise RuntimeError("You must call fit_eval or restore first before calling predict!")
        check_tf_version()
        f = K.function([self.model.layers[0].input, K.learning_phase()],
                       [self.model.layers[-1].output])
        result = np.array([f((x, 1))[0] for _ in range(n_iter)])

        prediction = result.mean(axis=0)
        uncertainty = result.var(axis=0)
        return prediction, uncertainty

    def state_dict(self):
        state = {
            "config": self.config,
            "weights": self.model.get_weights(),
            # "optimizer_weights": self.model.optimizer.get_weights()
        }
        return state

    def load_state_dict(self, state):
        self.config = state["config"]
        self.model = self.model_creator(self.config)
        self.model.set_weights(state["weights"])
        self.model_built = True
        # self.model.optimizer.set_weights(state["optimizer_weights"])

    def save(self, checkpoint_file, config_path=None):
        if not self.model_built:
            raise RuntimeError("You must call fit_eval or restore first before calling save!")
        state_dict = self.state_dict()
        with open(checkpoint_file, "wb") as f:
            pickle.dump(state_dict, f)

    def restore(self, checkpoint_file, **config):
        with open(checkpoint_file, "rb") as f:
            state_dict = pickle.load(f)
        self.load_state_dict(state_dict)

    def _get_required_parameters(self):
        return {"metric"}

    def _get_optional_parameters(self):
        return {"batch_size"}

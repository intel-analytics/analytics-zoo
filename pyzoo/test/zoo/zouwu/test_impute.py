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
import pandas as pd
import pytest

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
from zoo.zouwu.preprocessing.impute.LastFill import LastFill
from zoo.zouwu.preprocessing.impute.MeanFill import MeanFill
from zoo.zouwu.preprocessing.impute.MFFill import MFFill


class TestDataImputation(ZooTestCase):

    def setup_method(self, method):
        self.ft = TimeSequenceFeatureTransformer()
        self.create_data()

    def teardown_method(self, method):
        pass

    def create_data(self):
        data = np.random.random_sample((5, 50))
        mask = np.random.random_sample((5, 50))
        mask[mask >= 0.4] = 2
        mask[mask < 0.4] = 1
        mask[mask < 0.2] = 0
        data[mask == 0] = None
        data[mask == 1] = np.nan
        df = pd.DataFrame(data)
        idx = pd.date_range(start='2020-07-01 00:00:00', end='2020-07-01 08:00:00', freq='2H')
        df.index = idx
        self.data = df

    def test_lastfill(self):
        last_fill = LastFill()
        mse_missing = last_fill.evaluate(self.data, 0.1)
        imputed_data = last_fill.impute(self.data)
        assert imputed_data.isna().sum().sum() == 0
        mse = last_fill.evaluate(imputed_data, 0.1)

    def test_meanfill(self):
        mean_fill = MeanFill()
        mse_missing = mean_fill.evaluate(self.data, 0.1)
        imputed_data = mean_fill.impute(self.data)
        assert imputed_data.isna().sum().sum() == 0
        mse = mean_fill.evaluate(imputed_data, 0.1)

    def test_mffill(self):
        mf_fill = MFFill()
        mse_missing = mf_fill.evaluate(self.data, 0.1)
        imputed_data = mf_fill.impute(self.data)
        assert imputed_data.isna().sum().sum() == 0
        mse = mf_fill.evaluate(imputed_data, 0.1)


if __name__ == "__main__":
    pytest.main([__file__])

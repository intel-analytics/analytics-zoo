import pytest
import pandas as pd
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.automl.model.holt_winters import *
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
from numpy.testing import assert_array_almost_equal


class TestHoltWinters(ZooTestCase):

    def setup_method(self, method):
        # super().setup_method(method)
        
        #train_data = pd.DataFrame(data=np.random.randn(64, 4))
        #val_data = pd.DataFrame(data=np.random.randn(16, 4))
        #test_data = pd.DataFrame(data=np.random.randn(16, 4))
        dates = pd.date_range('20200101', periods=96)
        df = pd.DataFrame(np.random.randn(96), index=dates, columns=['value'])
        self.x_train = df[:64]
        self.y_train = df[:64]['value'].to_numpy()
        self.x_val = df[64:80]
        self.y_val = df[64:80]['value'].to_numpy()
        self.x_test = df[80:]
        self.y_test = df[80:]['value'].to_numpy()
        
        future_seq_len = 1

        # use roll method in time_sequence
        
        self.config = {
            'column_name': 'value',
            'future_seq_len': future_seq_len,
            'model_type': "['simple', 'add_add']"
        }
        self.model = HoltWinters(check_optional_config=False, future_seq_len=future_seq_len)
        
    def teardown_method(self, method):
        pass

    def test_fit_eval(self):
        print("fit_eval:", self.model.fit_eval(self.x_train,
                                               self.y_train,
                                               **self.config))

    def test_evaluate(self):
        self.model.fit_eval(self.x_train, self.y_train, **self.config)
        mse, rs = self.model.evaluate(self.x_val,
                                      self.y_val,
                                      metric=['mse', 'r2'])
        print("Mean squared error is:", mse)
        print("R square is:", rs)

    def test_predict(self):
        self.model.fit_eval(self.x_train, self.y_train, **self.config)
        self.y_pred = self.model.predict(self.x_test)
        assert self.y_pred.shape == (self.x_test.shape[0],)

    def test_save_restore(self):
        new_model = HoltWinters()
        self.model.fit_eval(self.x_train, self.y_train, **self.config)
        predict_before = self.model.predict(self.x_test)

        dirname = tempfile.mkdtemp(prefix="automl_test_holtwinters")
        filename = tempfile.mkstemp(suffix='.json', dir=dirname)[1]
        csvname = tempfile.mkstemp(suffix='.csv', dir=dirname)[1]
        try:
            self.model.save(filename, csvname)
            config = load_config(filename)
            new_model.restore(csvname, **config)
            predict_after = new_model.predict(self.x_test)
            assert_array_almost_equal(predict_before, predict_after, decimal=2)

        finally:
            shutil.rmtree(dirname)


if __name__ == '__main__':
    pytest.main([__file__])
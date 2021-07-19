import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO
import statsmodels
import json

from zoo.automl.model.abstract import BaseModel
from zoo.automl.common.util import *
from zoo.automl.common.metrics import Evaluator

class HoltWinters(BaseModel):
  
  def __init__(self, check_optional_config=False, future_seq_len=1):
    """
    Constructor of Holt_Winters model
    """
    self.model = None
    self.future_seq_len = future_seq_len
    self.model_type = None
    self.params = None
    self.column_name = None
    self.train = None
  
  def fit_eval(self, x, y, **config):
    """
    Fit the model and choose the best model type
    :param x: dataframe with timestamps and informations
    :param y: 1-d numpy array with true values
    :param config: hyper parameters
    :return:
    """
    super()._check_config(**config)
    self.column_name = config["column_name"]
    self.model_type = config.get('model_type', 'simple')
    self.train = x
    mod = statsmodels.tsa.holtwinters.ExponentialSmoothing(x[self.column_name])
    res = mod.fit()
    self.model = 'simple'
    self.params = res.params
    y_pred = res.predict(start=x.index[0], end=x.index[-1])
    print('y shape', y.shape)
    print('y_pred shape', y_pred.shape)
    evaluate = Evaluator.evaluate('mse', y, y_pred)
    if 'add_add' in self.model_type:
      mod = statsmodels.tsa.holtwinters.ExponentialSmoothing(x[self.column_name], trend='add', seasonal='add')
      res = mod.fit()
      y_pred = res.predict(start=x.index[0], end=x.index[-1])
      if Evaluator.evaluate('mse', y.transpose(), y_pred)<evaluate:
        self.model = 'add_add'
        self.params = res.params
        evaluate = Evaluator.evaluate('mse', y.transpose(), y_pred)
    
    if 'add_mul' in self.model_type:
      mod = statsmodels.tsa.holtwinters.ExponentialSmoothing(x[self.column_name], trend='add', seasonal='mul')
      res = mod.fit()
      y_pred = res.predict(start=x.index[0], end=x.index[-1])
      if Evaluator.evaluate('mse', y.transpose(), y_pred)<evaluate:
        self.model = 'add_mul'
        self.params = res.params
        evaluate = Evaluator.evaluate('mse', y.transpose(), y_pred)
  
  def evaluate(self, x, y, metric=['mse']):
    """
    Evaluate on x, y
    :param x: input
    :param y: target
    :param metric: a list of metrics in string format
    :return: a list of metric evaluation results
    """
    y_pred = self.predict(x)
    return [Evaluator.evaluate(m, y.transpose(), y_pred) for m in metric]
    
  def predict(self, x):
    """
    Prediction on x.
    :param x: input
    :return: predicted y
    """
    # date_diff = x.index[-1]-x.index[-2]
    # dates = []
    # for i in range(self.future_seq_len):
    #   dates.append(x.index[-1]+(i+1)*date_diff)
    # new_x = pd.DataFrame(np.zeros(self.future_seq_len*len(x.columns)).reshape(self.future_seq_len,len(x.columns)), index=dates, columns=x.columns)
    new_x = pd.DataFrame(np.zeros(len(x.index)*len(self.train.columns)).reshape(len(x.index), len(self.train.columns)), index=x.index, columns=self.train.columns)
    start_pos = len(self.train.index)
    x = self.train.append(new_x)
    if self.model=='simple':
      mod = statsmodels.tsa.holtwinters.ExponentialSmoothing(x[self.column_name])
      res = statsmodels.tsa.holtwinters.HoltWintersResults(mod, self.params)
      pred = res.predict(start=x.index[start_pos], end=x.index[-1])
    if self.model=='add_add':
      mod = statsmodels.tsa.holtwinters.ExponentialSmoothing(x[self.column_name], trend='add', seasonal='add')
      res = statsmodels.tsa.holtwinters.HoltWintersResults(mod, self.params)
      pred = res.predict(start=x.index[start_pos], end=x.index[-1])
    if self.model=='add_mul':
      mod = statsmodels.tsa.holtwinters.ExponentialSmoothing(x[self.column_name], trend='add', seasonal='mul')
      res = statsmodels.tsa.holtwinters.HoltWintersResults(mod, self.params)
      pred = res.predict(start=x.index[start_pos], end=x.index[-1])
    return pred
    
  def save(self, config_path, df_path):
    """
    save model hyper parameters.
    :param config_path: the config file
    :return:
    """
    config_to_save = {
      "model": self.model,
      "params": self.params,
      "column_name": self.column_name
    }
    self.train.to_csv(df_path, index=True)
    save_config(config_path, config_to_save, replace=True)
  
  def restore(self, df_path, **config):
    """
    restore model hyper parameters.
    :param config_path: the config file
    :return:
    """
    self.model = config['model']
    self.params = config['params']
    self.train = pd.read_csv(df_path)
    self.column_name = config['column_name']
    
  def _get_optional_parameters(self):
    return set(["model_type"])

  def _get_required_parameters(self):
    return set(["future_seq_len"])

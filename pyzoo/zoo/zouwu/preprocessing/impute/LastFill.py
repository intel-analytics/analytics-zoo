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
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from zoo.zouwu.preprocessing.impute.abstract import BaseImpute
from zoo.zouwu.preprocessing.impute.MFactorization import MF
from sklearn.preprocessing import MinMaxScaler

class LastFill(BaseImpute):
    """
    Impute missing data with last seen value
    """
    def __init__(self):
        """
        Construct model for last filling method
        """
        pass

    def impute(self, df):
        """
        impute data
        :params df: input dataframe
        :return: imputed dataframe
        """
        df.iloc[0] = df.iloc[0].fillna(0)
        return df.fillna(method='pad')
       
class MeanFill(BaseImpute):
    """
    Impute missing data with mean value
    """
    def __init__(self):
        """
        Construct model for mean filling method
        """
    pass

    def impute(self, df):
        """
        impute data
        :params df: input dataframe
        :return: imputed dataframe
        """
        mean = df.mean()
        return df.fillna(mean)

class MFFill(BaseImpute):
    """
    Impute missing data with MF value
    """
    def __init__(self):
        """
        Construct model for matrix factorization method
        """
    pass
    
    def scaling(self,x):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(x)
        scaled_x = scaler.transform(x)
        return scaled_x, scaler

    def inverse_scale(self,scaler, x):
        return scaler.inverse_transform(x)

    def impute(self, df, k=1, alpha=0.01, beta=0.1, iterations=100):
        """
        impute data
        :params df(dataframe): input dataframe
        :params k(int): number of latent dimensions
        :params alpha (float) : learning rate
        :params beta (float)  : regularization parameter
        """
        scaled_df,scaler = self.scaling(x=df.values)
        mf_df = pd.DataFrame(scaled_df)
        mf = MF(mf_df, k=k, alpha=alpha, beta=beta, iterations=iterations)
        mf.train()
        X_hat = mf.full_matrix()
        X_comp = mf.replace_nan(X_hat)
        filled_unscaled = self.inverse_scale(scaler,X_comp)
        filled_df = pd.DataFrame(filled_unscaled)
        filled_df.columns = df.columns
        filled_df.index = df.index
        return filled_df

class KNNFill(BaseImpute):
    """
    Impute missing data with k nearest neighbors value
    """
    def __init__(self):
        """
        Construct model for KNN method
        """
    pass
    
    def knn_missing_filled(self,x_train, y_train, test, k = 3, dispersed = False):
        if dispersed:
            clf = KNeighborsClassifier(n_neighbors = k, weights = "distance")
        else:
            clf = KNeighborsRegressor(n_neighbors = k, weights = "distance")
        clf.fit(x_train, y_train)
        return test.index, clf.predict(test)
    
    def train(self,df,i,col_name):
        train_y = df[df.isna()[col_name]==False][col_name]
        cols = df.columns.tolist()
        cols.pop(i)
        train_x = df[df.isna()[col_name]==False].loc[:,cols]
        train_x = train_x.fillna(train_x.mean())
        test_x = df[df.isna()[col_name]==True].loc[:,cols]
        test_x  = test_x.fillna(test_x.mean())
        test_cols = test_x.columns.tolist()
        for j in range(len(test_x.columns)):
            if test_x.isna()[test_cols[j]].sum()!=0:
                test_cols.pop(j)
        test_x = test_x.loc[:,test_cols]
        if test_x.empty:
            raise Exception("Input value is empty")
        test_index, pred_values = self.knn_missing_filled(train_x,train_y,test_x)
        return test_index, pred_values

    def impute(self, df):
        """
        impute data
        :params df: input dataframe
        :return: imputed dataframe
        """
        cols = df.columns
        for i in range(len(df.columns)):
            if df.isna()[cols[i]].sum() != 0:
                try:
                    test_index, pred_values = self.train(df,i,cols[i])
                    df.loc[test_index,cols[i]] = pred_values
                except (Exception):
                    print("Error: value error")
                    return
        return df
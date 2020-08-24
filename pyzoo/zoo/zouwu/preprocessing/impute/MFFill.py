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

from zoo.zouwu.preprocessing.impute.abstract import BaseImpute

from sklearn.preprocessing import MinMaxScaler


class MF():
    """
    Impute missing data with Matrix Factorization value
    """
    def __init__(self, df, k, alpha, beta, iterations):
        """
        Construct model for MF imputation method
        :params df(dataframe): input dataframe
        :params k(int): number of latent dimensions
        :params alpha (float) : learning rate
        :params beta (float)  : regularization parameter
        """
        self.X = df.values
        self.num_samples, self.num_features = self.X.shape
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        mask = (-df.isna())*1
        self.not_nan_index = mask.to_numpy()

    pass

    def train(self):
        # Initialize factorization matrix U and V
        self.U = np.random.normal(scale=1./self.k, size=(self.num_samples, self.k))
        self.V = np.random.normal(scale=1./self.k, size=(self.num_features, self.k))

        # Initialize the biases
        self.b_u = np.zeros(self.num_samples)
        self.b_v = np.zeros(self.num_features)
        self.b = np.mean(self.X[np.where(self.not_nan_index)])
        # Create a list of training samples
        self.samples = [
            (i, j, self.X[i, j])
            for i in range(self.num_samples)
            for j in range(self.num_features)
            if not np.isnan(self.X[i, j])
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            # total square error
            se = self.square_error()
            training_process.append((i, se))
            if (i+1) % 20 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, se))

        return training_process

    def square_error(self):
        """
        A function to compute the total square error
        """
        predicted = self.full_matrix()
        error = 0
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if self.not_nan_index[i, j]:
                    error += pow(self.X[i, j] - predicted[i, j], 2)
        return error

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, x in self.samples:
            # Computer prediction and error
            prediction = self.get_x(i, j)
            e = (x - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (2 * e - self.beta * self.b_u[i])
            self.b_v[j] += self.alpha * (2 * e - self.beta * self.b_v[j])

            # Update factorization matrix U and V
            """
            If RuntimeWarning: overflow encountered in multiply,
            then turn down the learning rate alpha.
            """
            self.U[i, :] += self.alpha * (2 * e * self.V[j, :] - self.beta * self.U[i, :])
            self.V[j, :] += self.alpha * (2 * e * self.U[i, :] - self.beta * self.V[j, :])

    def get_x(self, i, j):
        """
        Get the predicted x of sample i and feature j
        """
        prediction = self.b + self.b_u[i] + self.b_v[j] + self.U[i, :].dot(self.V[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, U and V
        """
        return self.b + self.b_u[:, np.newaxis] + self.b_v[np.newaxis, :] + self.U.dot(self.V.T)

    def replace_nan(self, X_hat):
        """
        Replace np.nan of X with the corresponding value of X_hat
        """
        X = np.copy(self.X)
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if np.isnan(X[i, j]):
                    X[i, j] = X_hat[i, j]
        return X


class MFFill(BaseImpute):
    """
    Impute missing data with MF value
    """
    def __init__(self):
        """
        Construct model for matrix factorization method
        """
    pass

    def scaling(self, x):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(x)
        scaled_x = scaler.transform(x)
        return scaled_x, scaler

    def inverse_scale(self, scaler, x):
        return scaler.inverse_transform(x)

    def impute(self, df, k=1, alpha=0.01, beta=0.1, iterations=100):
        """
        impute data
        :params df(dataframe): input dataframe
        :params k(int): number of latent dimensions
        :params alpha (float) : learning rate
        :params beta (float)  : regularization parameter
        """
        scaled_df, scaler = self.scaling(x=df.values)
        mf_df = pd.DataFrame(scaled_df)
        mf = MF(mf_df, k=k, alpha=alpha, beta=beta, iterations=iterations)
        mf.train()
        X_hat = mf.full_matrix()
        X_comp = mf.replace_nan(X_hat)
        filled_unscaled = self.inverse_scale(scaler, X_comp)
        filled_df = pd.DataFrame(filled_unscaled)
        filled_df.columns = df.columns
        filled_df.index = df.index
        print(filled_df.isna().sum().sum())
        return filled_df

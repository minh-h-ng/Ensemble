# -*- coding: utf-8 -*-

import base as forecast
import numpy as np
import pandas as pd
import tqdm
from sklearn.base import BaseEstimator


class AverageEstimator(BaseEstimator):
    def __init__(self):
        self.algo = forecast.ForecastAlgorithms(samples=500)

    def fit(self, X):
        self.fit_ = X  # # memorize data for prediction

    def predict(self, X):
        # X here holds the true observations

        # first prediction is done on already memorized data
        avg_prediction = np.average([
            self.algo.naive_forecast(self.fit_),
            self.algo.ar_forecast(self.fit_),
            self.algo.arma_forecast(self.fit_),
            self.algo.arima_forecast(self.fit_),
            self.algo.ets_forecast(self.fit_)
        ])
        predictions = np.array([
            avg_prediction
        ])

        # subsequent predictions are online
        for i in tqdm.tqdm(range(1, len(X))):
            data = pd.concat([self.fit_, X[:i]])  # # training + elapsed
            avg_prediction = np.average([
                self.algo.naive_forecast(data),
                self.algo.ar_forecast(data),
                self.algo.arma_forecast(data),
                self.algo.arima_forecast(data),
                self.algo.ets_forecast(data)
            ])
            predictions = np.append(predictions, avg_prediction)
        return predictions

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        numerator = np.absolute(predictions - observations)
        denominator = observations
        return sum(numerator / denominator)


class NaiveEstimator(BaseEstimator):
    def __init__(self):
        self.algo = forecast.ForecastAlgorithms(samples=500)

    def fit(self, X):
        self.fit_ = X  # # memorize data for prediction
        return self

    def predict(self, X):
        # X here holds the true observations

        # first prediction is done on already memorized data
        predictions = np.array([
            self.algo.naive_forecast(self.fit_)
        ])

        # subsequent predictions are online
        for i in tqdm.tqdm(range(1, len(X))):
            data = pd.concat([self.fit_, X[:i]])  # # training + elapsed
            predictions = np.append(predictions,
                                    self.algo.naive_forecast(data))
        return predictions

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        numerator = np.absolute(predictions - observations)
        denominator = observations
        return sum(numerator / denominator)


class ArEstimator(BaseEstimator):
    def __init__(self):
        self.algo = forecast.ForecastAlgorithms(samples=500)

    def fit(self, X):
        self.fit_ = X  # # memorize data for prediction
        return self

    def predict(self, X):
        # X here holds the true observations

        # first prediction is done on already memorized data
        predictions = np.array([
            self.algo.ar_forecast(self.fit_)
        ])

        # subsequent predictions are online
        for i in tqdm.tqdm(range(1, len(X))):
            data = pd.concat([self.fit_, X[:i]])  # # training + elapsed
            predictions = np.append(predictions,
                                    self.algo.ar_forecast(data))
        return predictions

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        numerator = np.absolute(predictions - observations)
        denominator = observations
        return sum(numerator / denominator)


class ArmaEstimator(BaseEstimator):
    def __init__(self):
        self.algo = forecast.ForecastAlgorithms(samples=500)

    def fit(self, X):
        self.fit_ = X  # # memorize data for prediction
        return self

    def predict(self, X):
        # X here holds the true observations

        # first prediction is done on already memorized data
        predictions = np.array([
            self.algo.arma_forecast(self.fit_)
        ])

        # subsequent predictions are online
        for i in tqdm.tqdm(range(1, len(X))):
            data = pd.concat([self.fit_, X[:i]])  # # training + elapsed
            predictions = np.append(predictions,
                                    self.algo.arma_forecast(data))
        return predictions

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        numerator = np.absolute(predictions - observations)
        denominator = observations
        return sum(numerator / denominator)


class ArimaEstimator(BaseEstimator):
    def __init__(self):
        self.algo = forecast.ForecastAlgorithms(samples=500)

    def fit(self, X):
        self.fit_ = X  # # memorize data for prediction
        return self

    def predict(self, X):
        # X here holds the true observations

        # first prediction is done on already memorized data
        predictions = np.array([
            self.algo.arima_forecast(self.fit_)
        ])

        # subsequent predictions are online
        for i in tqdm.tqdm(range(1, len(X))):
            data = pd.concat([self.fit_, X[:i]])  # # training + elapsed
            predictions = np.append(predictions,
                                    self.algo.arima_forecast(data))
        return predictions

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        numerator = np.absolute(predictions - observations)
        denominator = observations
        return sum(numerator / denominator)


class EtsEstimator(BaseEstimator):
    def __init__(self):
        self.algo = forecast.ForecastAlgorithms(samples=500)

    def fit(self, X):
        self.fit_ = X  # # memorize data for prediction
        return self

    def predict(self, X):
        # X here holds the true observations

        # first prediction is done on already memorized data
        predictions = np.array([
            self.algo.ets_forecast(self.fit_)
        ])

        # subsequent predictions are online
        for i in tqdm.tqdm(range(1, len(X))):
            data = pd.concat([self.fit_, X[:i]])  # # training + elapsed
            predictions = np.append(predictions,
                                    self.algo.ets_forecast(data))
        return predictions

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        numerator = np.absolute(predictions - observations)
        denominator = observations
        return sum(numerator / denominator)

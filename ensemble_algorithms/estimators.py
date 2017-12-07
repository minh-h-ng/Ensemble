# -*- coding: utf-8 -*-

import base as forecast
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


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
        for i in range(1, len(X)):
            data = pd.concat([self.fit_, X[:i]])  # # training + elapsed
            predictions = np.append(predictions,
                                    self.algo.naive_forecast(data))
        return predictions

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        minima = np.minimum(predictions, observations)
        maxima = np.maximum(predictions, observations)
        return sum(minima / maxima)


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
        for i in range(1, len(X)):
            data = pd.concat([self.fit_, X[:i]])  # # training + elapsed
            predictions = np.append(predictions,
                                    self.algo.ar_forecast(data))
        return predictions

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        minima = np.minimum(predictions, observations)
        maxima = np.maximum(predictions, observations)
        return sum(minima / maxima)


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
        for i in range(1, len(X)):
            data = pd.concat([self.fit_, X[:i]])  # # training + elapsed
            predictions = np.append(predictions,
                                    self.algo.arma_forecast(data))
        return predictions

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        minima = np.minimum(predictions, observations)
        maxima = np.maximum(predictions, observations)
        return sum(minima / maxima)


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
        for i in range(1, len(X)):
            data = pd.concat([self.fit_, X[:i]])  # # training + elapsed
            predictions = np.append(predictions,
                                    self.algo.arima_forecast(data))
        return predictions

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        minima = np.minimum(predictions, observations)
        maxima = np.maximum(predictions, observations)
        return sum(minima / maxima)


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
        for i in range(1, len(X)):
            data = pd.concat([self.fit_, X[:i]])  # # training + elapsed
            predictions = np.append(predictions,
                                    self.algo.ets_forecast(data))
        return predictions

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        minima = np.minimum(predictions, observations)
        maxima = np.maximum(predictions, observations)
        return sum(minima / maxima)

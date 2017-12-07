# -*- coding: utf-8 -*-

import base as forecast
import numpy as np
from sklearn.base import BaseEstimator


class NaiveEstimator(BaseEstimator):
    def __init__(self):
        self.algo = forecast.ForecastAlgorithms(samples=500)

    def fit(self, X):
        self.fit_ = X  # # memorize data for prediction
        return self

    def predict(self, X):
        # X here holds the true observations,
        # do not use the values for forecasting.
        # It should only be used to count how many forecasts
        # are requested
        xlen = len(X)  # # number of predictions
        return self.algo.naive_forecast(self.fit_, n=xlen)

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
        # X here holds the true observations,
        # do not use the values for forecasting.
        # It should only be used to count how many forecasts
        # are requested
        xlen = len(X)  # # number of predictions
        return self.algo.ar_forecast(self.fit_, n=xlen)

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
        # X here holds the true observations,
        # do not use the values for forecasting.
        # It should only be used to count how many forecasts
        # are requested
        xlen = len(X)  # # number of predictions
        return self.algo.arma_forecast(self.fit_, n=xlen)

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
        # X here holds the true observations,
        # do not use the values for forecasting.
        # It should only be used to count how many forecasts
        # are requested
        xlen = len(X)  # # number of predictions
        return self.algo.arima_forecast(self.fit_, n=xlen)

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
        # X here holds the true observations,
        # do not use the values for forecasting.
        # It should only be used to count how many forecasts
        # are requested
        xlen = len(X)  # # number of predictions
        return self.algo.ets_forecast(self.fit_, n=xlen)

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        minima = np.minimum(predictions, observations)
        maxima = np.maximum(predictions, observations)
        return sum(minima / maxima)

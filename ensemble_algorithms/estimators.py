# -*- coding: utf-8 -*-

import base as forecast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.base import BaseEstimator
from torch.autograd import Variable


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
        return np.rint(predictions)

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


class NaiveBaggingEstimator(BaseEstimator):
    def __init__(self, nbags=10, samples_frac=0.6):
        # nbags = 10
        # samples_frac = 60% of training samples
        self.nbags = nbags
        self.samples_frac = samples_frac
        self.algo = forecast.ForecastAlgorithms(samples=500)

    def fit(self, X):
        # X here is the whole training set
        # we have to randomly sample
        # and calculate average predictions
        self.samples_ = [None for _ in range(self.nbags)]

        for bag in range(self.nbags):
            # memorize samples
            self.samples_[bag] = X.sample(
                frac=self.samples_frac
            ).sort_index()

        return self

    def predict(self, X):
        # X here holds the true observations

        # first prediction is done on already memorized data
        predictions = np.array([])
        bag_predictions = np.array([])
        for bag in range(self.nbags):
            # Figure out step size by calculating how many
            # steps X is ahead of bagged samples
            step = int(
                (X.index[0] - self.samples_[bag].index[-1])
                / pd.Timedelta('1 hour')
            )

            # Get the last output (that will be the prediction step size ahead)
            pred = self.algo.naive_forecast(data=self.samples_[bag], n=step)[-1]
            bag_predictions = np.append(bag_predictions, pred)

        # mean
        assert len(bag_predictions) == self.nbags
        avg = np.average(bag_predictions)
        predictions = np.append(predictions, avg)

        # subsequent predictions are online
        # so it will always be a 1-step process
        for i in tqdm.tqdm(range(1, len(X))):
            bag_predictions = np.array([])
            for bag in range(self.nbags):
                # training + elapsed
                data = pd.concat([self.samples_[bag], X[:i]])
                step = int(
                    (X.index[i] - data.index[-1])
                    / pd.Timedelta('1 hour')
                )
                assert step == 1
                bag_predictions = np.append(bag_predictions,
                                            self.algo.naive_forecast(data))
            # mean
            assert len(bag_predictions) == self.nbags
            avg = np.average(bag_predictions)
            predictions = np.append(predictions, avg)

        return np.rint(predictions)

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        numerator = np.absolute(predictions - observations)
        denominator = observations
        return sum(numerator / denominator)


class ArBaggingEstimator(BaseEstimator):
    def __init__(self, nbags=10, samples_frac=0.6):
        # nbags = 10
        # samples_frac = 60% of training samples
        self.nbags = nbags
        self.samples_frac = samples_frac
        self.algo = forecast.ForecastAlgorithms(samples=500)

    def fit(self, X):
        # X here is the whole training set
        # we have to randomly sample
        # and calculate average predictions
        self.samples_ = [None for _ in range(self.nbags)]

        for bag in range(self.nbags):
            # memorize samples
            self.samples_[bag] = X.sample(
                frac=self.samples_frac
            ).sort_index()

        return self

    def predict(self, X):
        # X here holds the true observations

        # first prediction is done on already memorized data
        predictions = np.array([])
        bag_predictions = np.array([])
        for bag in range(self.nbags):
            # Figure out step size by calculating how many
            # steps X is ahead of bagged samples
            step = int(
                (X.index[0] - self.samples_[bag].index[-1])
                / pd.Timedelta('1 hour')
            )

            # Get the last output (that will be the prediction step size ahead)
            pred = self.algo.ar_forecast(data=self.samples_[bag], n=step)[-1]
            bag_predictions = np.append(bag_predictions, pred)

        # mean
        assert len(bag_predictions) == self.nbags
        avg = np.average(bag_predictions)
        predictions = np.append(predictions, avg)

        # subsequent predictions are online
        # so it will always be a 1-step process
        for i in tqdm.tqdm(range(1, len(X))):
            bag_predictions = np.array([])
            for bag in range(self.nbags):
                # training + elapsed
                data = pd.concat([self.samples_[bag], X[:i]])
                step = int(
                    (X.index[i] - data.index[-1])
                    / pd.Timedelta('1 hour')
                )
                assert step == 1
                bag_predictions = np.append(bag_predictions,
                                            self.algo.ar_forecast(data))
            # mean
            assert len(bag_predictions) == self.nbags
            avg = np.average(bag_predictions)
            predictions = np.append(predictions, avg)

        return np.rint(predictions)

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        numerator = np.absolute(predictions - observations)
        denominator = observations
        return sum(numerator / denominator)


class ArmaBaggingEstimator(BaseEstimator):
    def __init__(self, nbags=10, samples_frac=0.6):
        # nbags = 10
        # samples_frac = 60% of training samples
        self.nbags = nbags
        self.samples_frac = samples_frac
        self.algo = forecast.ForecastAlgorithms(samples=500)

    def fit(self, X):
        # X here is the whole training set
        # we have to randomly sample
        # and calculate average predictions
        self.samples_ = [None for _ in range(self.nbags)]

        for bag in range(self.nbags):
            # memorize samples
            self.samples_[bag] = X.sample(
                frac=self.samples_frac
            ).sort_index()

        return self

    def predict(self, X):
        # X here holds the true observations

        # first prediction is done on already memorized data
        predictions = np.array([])
        bag_predictions = np.array([])
        for bag in range(self.nbags):
            # Figure out step size by calculating how many
            # steps X is ahead of bagged samples
            step = int(
                (X.index[0] - self.samples_[bag].index[-1])
                / pd.Timedelta('1 hour')
            )

            # Get the last output (that will be the prediction step size ahead)
            pred = self.algo.arma_forecast(data=self.samples_[bag], n=step)[-1]
            bag_predictions = np.append(bag_predictions, pred)

        # mean
        assert len(bag_predictions) == self.nbags
        avg = np.average(bag_predictions)
        predictions = np.append(predictions, avg)

        # subsequent predictions are online
        # so it will always be a 1-step process
        for i in tqdm.tqdm(range(1, len(X))):
            bag_predictions = np.array([])
            for bag in range(self.nbags):
                # training + elapsed
                data = pd.concat([self.samples_[bag], X[:i]])
                step = int(
                    (X.index[i] - data.index[-1])
                    / pd.Timedelta('1 hour')
                )
                assert step == 1
                bag_predictions = np.append(bag_predictions,
                                            self.algo.arma_forecast(data))
            # mean
            assert len(bag_predictions) == self.nbags
            avg = np.average(bag_predictions)
            predictions = np.append(predictions, avg)

        return np.rint(predictions)

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        numerator = np.absolute(predictions - observations)
        denominator = observations
        return sum(numerator / denominator)


class ArimaBaggingEstimator(BaseEstimator):
    def __init__(self, nbags=10, samples_frac=0.6):
        # nbags = 10
        # samples_frac = 60% of training samples
        self.nbags = nbags
        self.samples_frac = samples_frac
        self.algo = forecast.ForecastAlgorithms(samples=500)

    def fit(self, X):
        # X here is the whole training set
        # we have to randomly sample
        # and calculate average predictions
        self.samples_ = [None for _ in range(self.nbags)]

        for bag in range(self.nbags):
            # memorize samples
            self.samples_[bag] = X.sample(
                frac=self.samples_frac
            ).sort_index()

        return self

    def predict(self, X):
        # X here holds the true observations

        # first prediction is done on already memorized data
        predictions = np.array([])
        bag_predictions = np.array([])
        for bag in range(self.nbags):
            # Figure out step size by calculating how many
            # steps X is ahead of bagged samples
            step = int(
                (X.index[0] - self.samples_[bag].index[-1])
                / pd.Timedelta('1 hour')
            )

            # Get the last output (that will be the prediction step size ahead)
            pred = self.algo.arima_forecast(data=self.samples_[bag], n=step)[-1]
            bag_predictions = np.append(bag_predictions, pred)

        # mean
        assert len(bag_predictions) == self.nbags
        avg = np.average(bag_predictions)
        predictions = np.append(predictions, avg)

        # subsequent predictions are online
        # so it will always be a 1-step process
        for i in tqdm.tqdm(range(1, len(X))):
            bag_predictions = np.array([])
            for bag in range(self.nbags):
                # training + elapsed
                data = pd.concat([self.samples_[bag], X[:i]])
                step = int(
                    (X.index[i] - data.index[-1])
                    / pd.Timedelta('1 hour')
                )
                assert step == 1
                bag_predictions = np.append(bag_predictions,
                                            self.algo.arima_forecast(data))
            # mean
            assert len(bag_predictions) == self.nbags
            avg = np.average(bag_predictions)
            predictions = np.append(predictions, avg)

        return np.rint(predictions)

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        numerator = np.absolute(predictions - observations)
        denominator = observations
        return sum(numerator / denominator)


class EtsBaggingEstimator(BaseEstimator):
    def __init__(self, nbags=10, samples_frac=0.6):
        # nbags = 10
        # samples_frac = 60% of training samples
        self.nbags = nbags
        self.samples_frac = samples_frac
        self.algo = forecast.ForecastAlgorithms(samples=500)

    def fit(self, X):
        # X here is the whole training set
        # we have to randomly sample
        # and calculate average predictions
        self.samples_ = [None for _ in range(self.nbags)]

        for bag in range(self.nbags):
            # memorize samples
            self.samples_[bag] = X.sample(
                frac=self.samples_frac
            ).sort_index()

        return self

    def predict(self, X):
        # X here holds the true observations

        # first prediction is done on already memorized data
        predictions = np.array([])
        bag_predictions = np.array([])
        for bag in range(self.nbags):
            # Figure out step size by calculating how many
            # steps X is ahead of bagged samples
            step = int(
                (X.index[0] - self.samples_[bag].index[-1])
                / pd.Timedelta('1 hour')
            )

            # Get the last output (that will be the prediction step size ahead)
            pred = self.algo.ets_forecast(data=self.samples_[bag], n=step)[-1]
            bag_predictions = np.append(bag_predictions, pred)

        # mean
        assert len(bag_predictions) == self.nbags
        avg = np.average(bag_predictions)
        predictions = np.append(predictions, avg)

        # subsequent predictions are online
        # so it will always be a 1-step process
        for i in tqdm.tqdm(range(1, len(X))):
            bag_predictions = np.array([])
            for bag in range(self.nbags):
                # training + elapsed
                data = pd.concat([self.samples_[bag], X[:i]])
                step = int(
                    (X.index[i] - data.index[-1])
                    / pd.Timedelta('1 hour')
                )
                assert step == 1
                bag_predictions = np.append(bag_predictions,
                                            self.algo.ets_forecast(data))
            # mean
            assert len(bag_predictions) == self.nbags
            avg = np.average(bag_predictions)
            predictions = np.append(predictions, avg)

        return np.rint(predictions)

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        numerator = np.absolute(predictions - observations)
        denominator = observations
        return sum(numerator / denominator)


# Network for NeuralNetworkEstimator
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(5, 100)
        self.linear2 = torch.nn.Linear(100, 1)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class NeuralNetworkEstimator(BaseEstimator):
    def __init__(self, iterations=1000):
        self.iterations = iterations
        self.algo = forecast.ForecastAlgorithms(samples=500)

    def fit(self, X):
        # X here is the whole training set
        # run component algorithms
        components = np.empty((0, 5), np.float64)
        observations = np.empty((0, 1), np.float64)

        for i in tqdm.tqdm(range(1, len(X)), desc="Computing components"):
            naive = self.algo.naive_forecast(X[:i])[-1]
            ar = self.algo.ar_forecast(X[:i])[-1]
            arma = self.algo.arma_forecast(X[:i])[-1]
            arima = self.algo.arima_forecast(X[:i])[-1]
            ets = self.algo.ets_forecast(X[:i])[-1]

            components = np.append(components,
                                   np.array([[naive, ar, arma, arima, ets]]), axis=0)
            observations = np.append(observations,
                                     np.array([[X[i]]]), axis=0)

        self.model = Net()  # # network
        criterion = torch.nn.MSELoss(size_average=False)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4)

        # train the network for 1000 iterations
        for t in range(self.iterations):
            x = Variable(torch.from_numpy(components)).float()
            y = Variable(torch.from_numpy(observations), requires_grad=False).float()

            # forward pass
            y_pred = self.model(x)

            # backward pass
            loss = criterion(y_pred, y)
            print(t, loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, X):
        # X here holds the true observations
        pass

    def score(self, X):
        predictions = self.predict(X)
        observations = X.values

        numerator = np.absolute(predictions - observations)
        denominator = observations
        return sum(numerator / denominator)

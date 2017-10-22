# -*- coding: utf-8 -*-

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic


class ForecastAlgorithms:
    def __init__(self, file_path):
        """
        Initializes data required by forecasting algorithms
        :param file_path: path to processed dataset
        """
        self.series = series = pd.read_csv(
            file_path,
            header=None,  # contains no header
            index_col=0,  # set datetime column as index
            names=['datetime', 'requests'],  # name the columns
            converters={'datetime':  # custom datetime parser
                            lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')},
            squeeze=True,  # convert to Series
            dtype={'requests': np.float64}  # https://git.io/vdbyk
        )

    def naive_forecast(self, n=1):
        """
        Forecasts number of requests using naive algorithm
        :param n: number of out-of-sample hours for which forecast is requested
        :return: forecasts for all n hours
        """
        last_index = self.series.last_valid_index()
        return np.repeat(self.series[last_index], n)

    def ar_forecast(self, n=1):
        """
        Forecasts number of requests using AR(1) model
        :param n: number of out-of-sample hours for which forecast is requested
        :return: forecasts for all n hours
        """
        ar = ARMA(self.series, order=(1, 0))
        ar_fit = ar.fit(disp=0)
        return ar_fit.forecast(n)[0]

    def arma_forecast(self, n=1):
        """
        Forecasts number of requests using ARMA(1,1) model
        :param n: number of out-of-sample hours for which forecast is requested
        :return: forecasts for all n hours
        """
        arma = ARMA(self.series, order=(1, 1))
        arma_fit = arma.fit(disp=0)
        return arma_fit.forecast(n)[0]

    def arima_forecast(self, n=1):
        """
        Forecasts number of requests using ARIMA(p,d,q) model.
        The parameters (p,d,q) are auto-tuned.
        :param n: number of out-of-sample hours for which forecast is requested
        :return: forecasts for all n hours
        """
        # Test for stationarity
        dftest = adfuller(self.series, autolag='AIC')
        assert dftest[0] < dftest[4]['5%']  # Test Statistic < Critical Value (5%)
        assert dftest[1] < 0.05  # p-value < 0.05

        # Select p and q
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            arma_params = arma_order_select_ic(self.series, fit_kw=dict(method='css'))

        p = arma_params.bic_min_order[0]
        q = arma_params.bic_min_order[1]

        arima = ARIMA(self.series, order=(p, 0, q))
        arima_fit = arima.fit(disp=0)

        return arima_fit.forecast(n)[0]

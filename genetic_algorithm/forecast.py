# -*- coding: utf-8 -*-

import os
import warnings

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import tqdm
from rpy2.rinterface import RRuntimeError
from rpy2.robjects import pandas2ri


class ForecastAlgorithms:
    def __init__(self, samples=500):
        """
        Initializes forecasting algorithms
        :param samples: clip algorithms to most recent 500 samples
        """
        # Try importing 'forecast' package
        try:
            self.rforecast = rpackages.importr('forecast')
        except RRuntimeError:
            # Select mirror
            utils = rpackages.importr('utils')
            utils.chooseCRANmirror(ind=1)
            # Install
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                utils.install_packages('forecast')
            self.rforecast = rpackages.importr('forecast')

        # R timeseries
        self.rts = robjects.r('ts')

        # Clip
        if samples <= 0:
            raise ValueError
        self.clip = samples

    def naive_forecast(self, data, n=1):
        """
        Forecasts number of requests using naive algorithm
        :return: forecasts for next n hours
        """
        assert isinstance(data, pd.core.series.Series)

        # The last observed value will repeat as it is
        return np.repeat(data.values[-1], n)

    def ar_forecast(self, data, n=1):
        """
        Forecasts number of requests using AR(1) model
        :return: forecasts for next n hours
        """
        assert isinstance(data, pd.core.series.Series)

        pandas2ri.activate()
        results = np.array([])

        # series length
        series_len = len(data)

        # clip until 500
        if series_len > self.clip:
            start_idx = series_len - self.clip
            sub_series = data[start_idx:series_len]
        else:
            sub_series = data[:series_len]

        # forecast for next n hours
        if series_len <= 2:
            results = np.append(results, sub_series.mean())
        else:
            rdata = self.rts(sub_series)
            fit = self.rforecast.Arima(rdata, robjects.FloatVector((1, 0, 0)), method="ML")
            forecast = self.rforecast.forecast(fit, h=n)
            results = np.append(results, np.asarray(forecast[3]))

        return np.rint(results)

    def arma_forecast(self, data, n=1):
        """
        Forecasts number of requests using ARMA(1,1) model
        :return: forecasts for next n hours
        """
        assert isinstance(data, pd.core.series.Series)

        pandas2ri.activate()
        results = np.array([])

        # series length
        series_len = len(data)

        # clip until 500
        if series_len > self.clip:
            start_idx = series_len - self.clip
            sub_series = data[start_idx:series_len]
        else:
            sub_series = data[:series_len]

        # forecast for next n hours
        if series_len <= 2:
            results = np.append(results, sub_series.mean())
        else:
            rdata = self.rts(sub_series)
            fit = self.rforecast.Arima(rdata, robjects.FloatVector((1, 0, 1)), method="ML")
            forecast = self.rforecast.forecast(fit, h=n)
            results = np.append(results, np.asarray(forecast[3]))

        return np.rint(results)

    def arima_forecast(self, data, n=1):
        """
        Forecasts number of requests using ARIMA(p,d,q) model.
        The parameters (p,d,q) are auto-tuned.
        :return: forecasts for next n hours
        """
        assert isinstance(data, pd.core.series.Series)

        pandas2ri.activate()
        results = np.array([])

        # series length
        series_len = len(data)

        # clip until 500
        if series_len > self.clip:
            start_idx = series_len - self.clip
            sub_series = data[start_idx:series_len]
        else:
            sub_series = data[:series_len]

        # forecast for next n hours
        if series_len <= 2:
            results = np.append(results, sub_series.mean())
        else:
            rdata = self.rts(sub_series)
            fit = self.rforecast.auto_arima(rdata)
            forecast = self.rforecast.forecast(fit, h=n)
            results = np.append(results, np.asarray(forecast[3]))

        return np.rint(results)

    def ets_forecast(self, data, n=1):
        """
        Forecasts number of requests using ETS model.
        :return: forecasts for next n hours
        """
        assert isinstance(data, pd.core.series.Series)

        pandas2ri.activate()
        results = np.array([])

        # series length
        series_len = len(data)

        # clip until 500
        if series_len > self.clip:
            start_idx = series_len - self.clip
            sub_series = data[start_idx:series_len]
        else:
            sub_series = data[:series_len]

        # forecast for next n hours
        if series_len <= 2:
            results = np.append(results, sub_series.mean())
        else:
            rdata = self.rts(sub_series)
            fit = self.rforecast.ets(rdata)
            forecast = self.rforecast.forecast(fit, h=n)
            results = np.append(results, np.asarray(forecast[1]))

        return np.rint(results)


if __name__ == '__main__':
    # script's directory
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # read, write files
    dataPath = os.path.join(dir_path, '..', 'processed', 'cran_08_10.csv')
    writePath = os.path.join(dir_path, '..', 'PythonESN', 'data_backup', 'cran_08_10')

    # read csv
    series = pd.read_csv(
        dataPath,
        header=None,  # contains no header
        index_col=0,  # set datetime column as index
        names=['datetime', 'requests'],  # name the columns
        converters={'datetime':  # custom datetime parser
                        lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')},
        squeeze=True,  # convert to Series
        dtype={'requests': np.float64}  # https://git.io/vdbyk
    )

    # results (prediction at time 0 is invalid)
    naive_results = np.array([np.nan])
    ar_results = np.array([np.nan])
    arma_results = np.array([np.nan])
    arima_results = np.array([np.nan])
    ets_results = np.array([np.nan])

    # initialize algos
    algo = ForecastAlgorithms(samples=500)

    # simulate forecast for each elapsed hour
    # skip running for last hour
    # since there's no real observation to compare with
    for hr in tqdm.tqdm(range(1, len(series))):
        hr_data = series[:hr]

        # naive
        naive_results = np.append(naive_results,
                                  algo.naive_forecast(hr_data))
        # ar
        ar_results = np.append(ar_results,
                               algo.ar_forecast(hr_data))
        # arma
        arma_results = np.append(arma_results,
                                 algo.arma_forecast(hr_data))
        # arima
        arima_results = np.append(arima_results,
                                  algo.arima_forecast(hr_data))
        # ets
        ets_results = np.append(ets_results,
                                algo.ets_forecast(hr_data))

    # replace < 0 with 0
    naive_results[1:][naive_results[1:] < 0] = 0
    ar_results[1:][ar_results[1:] < 0] = 0
    arma_results[1:][arma_results[1:] < 0] = 0
    arima_results[1:][arima_results[1:] < 0] = 0
    ets_results[1:][ets_results[1:] < 0] = 0

    with open(writePath, 'w') as f:
        # Header
        line = "Naive,AR,ARMA,ARIMA,ETS,PreviousObservation,CurrentObservation"
        f.write(line + '\n')

        # Contents
        for i in range(1, len(series)):
            line = str(naive_results[i]) + ',' + str(ar_results[i]) + ',' \
                   + str(arma_results[i]) + ',' + str(arima_results[i]) + ',' \
                   + str(ets_results[i]) + ',' + str(series[i - 1]) + ',' + str(series[i])
            f.write(line + '\n')

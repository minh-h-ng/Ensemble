# -*- coding: utf-8 -*-

import warnings
import os

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import tqdm
from rpy2.rinterface import RRuntimeError
from rpy2.robjects import pandas2ri


class ForecastAlgorithms:
    def __init__(self, file_path):
        """
        Initializes data required by forecasting algorithms
        :param file_path: path to processed dataset
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

        # Read csv
        self.series = pd.read_csv(
            file_path,
            header=None,  # contains no header
            index_col=0,  # set datetime column as index
            names=['datetime', 'requests'],  # name the columns
            converters={'datetime':  # custom datetime parser
                            lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')},
            squeeze=True,  # convert to Series
            dtype={'requests': np.float64}  # https://git.io/vdbyk
        )

        # R timeseries
        self.rts = robjects.r('ts')

    def naive_simulation(self):
        """
        Forecasts number of requests using naive algorithm
        :return: forecasts for all hours
        """
        return np.append(np.nan, self.series.values[:-1])

    def ar_simulation(self):
        """
        Forecasts number of requests using AR(1) model
        :return: forecasts for all n hours
        """
        pandas2ri.activate()
        results = np.array([np.nan])

        # TODO: parallelize
        for i in tqdm.tqdm(range(self.series.size - 1)):
            if i>=500:
                sub_series = self.series[(i-499):(i+1)]
            else:
                sub_series = self.series[:(i + 1)]
            if i < 2:
                results = np.append(results, sub_series.mean())
            else:
                rdata = self.rts(sub_series)
                ar_fit = self.rforecast.Arima(rdata, robjects.FloatVector((1, 0, 0)), method="ML")
                ar_forecast = self.rforecast.forecast(ar_fit, h=1)
                results = np.append(results, ar_forecast[3])
        #return np.ceil(results)
        return np.rint(results)

    def arma_simulation(self):
        """
        Forecasts number of requests using ARMA(1,1) model
        :return: forecasts for all n hours
        """
        pandas2ri.activate()
        results = np.array([np.nan])

        # TODO: parallelize
        for i in tqdm.tqdm(range(self.series.size - 1)):
            if i>=500:
                sub_series = self.series[(i-499):(i+1)]
            else:
                sub_series = self.series[:(i + 1)]
            if i < 2:
                results = np.append(results, sub_series.mean())
            else:
                rdata = self.rts(sub_series)
                arma_fit = self.rforecast.Arima(rdata, robjects.FloatVector((1, 0, 1)), method="ML")
                arma_forecast = self.rforecast.forecast(arma_fit, h=1)
                results = np.append(results, arma_forecast[3])
        #return np.ceil(results)
        return np.rint(results)

    def arima_simulation(self):
        """
        Forecasts number of requests using ARIMA(p,d,q) model.
        The parameters (p,d,q) are auto-tuned.
        :return: forecasts for all n hours
        """
        pandas2ri.activate()
        results = np.array([np.nan])

        # TODO: parallelize
        for i in tqdm.tqdm(range(self.series.size - 1)):
            if i>=500:
                sub_series = self.series[(i-499):(i+1)]
            else:
                sub_series = self.series[:(i + 1)]
            if i < 2:
                results = np.append(results, sub_series.mean())
            else:
                rdata = self.rts(sub_series)
                arima_fit = self.rforecast.auto_arima(rdata)
                arima_forecast = self.rforecast.forecast(arima_fit, h=1)
                results = np.append(results, arima_forecast[3])
        #return np.ceil(results)
        return np.rint(results)

    def ets_simulation(self):
        """
        Forecasts number of requests using ETS model.
        :return: forecasts for all n hours
        """
        pandas2ri.activate()
        results = np.array([np.nan])

        # TODO: parallelize
        for i in tqdm.tqdm(range(self.series.size - 1)):
            if i>=500:
                sub_series = self.series[(i-499):(i+1)]
            else:
                sub_series = self.series[:(i + 1)]
            if i < 3:
                results = np.append(results, sub_series.mean())
            else:
                rdata = self.rts(sub_series)
                ets_fit = self.rforecast.ets(rdata)
                ets_forecast = self.rforecast.forecast(ets_fit, h=1)
                results = np.append(results, ets_forecast[1])
        #return np.ceil(results)
        return np.rint(results)

curPath = os.getcwd().split('/')
dataPath = ''
for i in range(len(curPath)-1):
    dataPath += curPath[i]+'/'
dataPath += 'processed/edgar.csv'

writePath = ''
for i in range(len(curPath)-1):
    writePath += curPath[i]+'/'
writePath += 'PythonESN/data/edgar'

forecast = ForecastAlgorithms(dataPath)
naive_results = forecast.naive_simulation()
ar_results = forecast.ar_simulation()
arma_results = forecast.arma_simulation()
arima_results = forecast.arma_simulation()
ets_results = forecast.ets_simulation()

with open(writePath,'w') as f:
    for i in range(1,len(naive_results)):
        line = str(naive_results[i]) + ',' + str(ar_results[i]) + ',' \
               + str(arma_results[i]) + ',' + str(arima_results[i]) + ',' \
               + str(ets_results[i]) + ',' + str(forecast.series[i])
        #line = str(naive_results[i])
        f.write(line+'\n')
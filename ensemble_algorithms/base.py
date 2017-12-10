# -*- coding: utf-8 -*-
import argparse
import collections
import warnings

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import tqdm
from rpy2.rinterface import RRuntimeError
from rpy2.robjects import pandas2ri


class ForecastAlgorithms:
    # Try importing 'forecast' package
    try:
        rforecast = rpackages.importr('forecast')
    except RRuntimeError:
        # Select mirror
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)
        # Install
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            utils.install_packages('forecast')
        rforecast = rpackages.importr('forecast')

    # R timeseries
    rts = robjects.r('ts')

    def __init__(self, samples=500):
        """
        Initializes forecasting algorithms
        :param samples: clip algorithms to use at-most past 500 samples
        """
        # Clip
        if samples <= 0:
            raise ValueError
        self.clip = samples

    def naive_forecast(self, data, n=1):
        """
        Forecasts number of requests using naive algorithm
        :param data: pandas Series object representing
                    data for already elapsed hours
        :param n: number of hours for which forecast is requested
        :return: forecasts for next n hours
        """
        assert isinstance(data, pd.core.series.Series)

        # The last observed value will repeat as-is
        return np.repeat(data.values[-1], n)

    def ar_forecast(self, data, n=1):
        """
        Forecasts number of requests using AR(1) model
        :param data: pandas Series object representing
                    data for already elapsed hours
        :param n: number of hours for which forecast is requested
        :return: forecasts for next n hours
        """
        assert isinstance(data, pd.core.series.Series)
        assert n >= 1

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
            # need at-least 3 true observations for using AR model
            # so, just calculate average if number of observations < 3
            results = np.append(results, sub_series.mean())

            # in such cases, it's hard to forecast for n > 1,
            # so sanitize!
            if n != 1:
                raise ValueError
        else:
            rdata = ForecastAlgorithms.rts(sub_series)
            fit = ForecastAlgorithms.rforecast.Arima(rdata,
                                                     robjects.FloatVector((1, 0, 0)),
                                                     method="ML")
            forecast = ForecastAlgorithms.rforecast.forecast(fit, h=n)
            results = np.append(results, np.asarray(forecast[3]))

        return np.rint(results)

    def arma_forecast(self, data, n=1):
        """
        Forecasts number of requests using ARMA(1,1) model
        :param data: pandas Series object representing
                    data for already elapsed hours
        :param n: number of hours for which forecast is requested
        :return: forecasts for next n hours
        """
        assert isinstance(data, pd.core.series.Series)
        assert n >= 1

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
            # need at-least 3 true observations for using ARMA model
            # so, just calculate average if number of observations < 3
            results = np.append(results, sub_series.mean())

            # in such cases, it's hard to forecast for n > 1,
            # so sanitize!
            if n != 1:
                raise ValueError
        else:
            rdata = ForecastAlgorithms.rts(sub_series)
            fit = ForecastAlgorithms.rforecast.Arima(rdata,
                                                     robjects.FloatVector((1, 0, 1)),
                                                     method="ML")
            forecast = ForecastAlgorithms.rforecast.forecast(fit, h=n)
            results = np.append(results, np.asarray(forecast[3]))

        return np.rint(results)

    def arima_forecast(self, data, n=1):
        """
        Forecasts number of requests using ARIMA(p,d,q) model.
        The parameters (p,d,q) are auto-tuned.
        :param data: pandas Series object representing
                    data for already elapsed hours
        :param n: number of hours for which forecast is requested
        :return: forecasts for next n hours
        """
        assert isinstance(data, pd.core.series.Series)
        assert n >= 1

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
            # need at-least 3 true observations for using ARIMA model
            # so, just calculate average if number of observations < 3
            results = np.append(results, sub_series.mean())

            # in such cases, it's hard to forecast for n > 1,
            # so sanitize!
            if n != 1:
                raise ValueError
        else:
            rdata = ForecastAlgorithms.rts(sub_series)
            fit = ForecastAlgorithms.rforecast.auto_arima(rdata)  # # auto fit
            forecast = ForecastAlgorithms.rforecast.forecast(fit, h=n)
            results = np.append(results, np.asarray(forecast[3]))

        return np.rint(results)

    def ets_forecast(self, data, n=1):
        """
        Forecasts number of requests using ETS model.
        :param data: pandas Series object representing
                    data for already elapsed hours
        :param n: number of hours for which forecast is requested
        :return: forecasts for next n hours
        """
        assert isinstance(data, pd.core.series.Series)
        assert n >= 1

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
            # need at-least 3 true observations for using ETS model
            # so, just calculate average if number of observations < 3
            results = np.append(results, sub_series.mean())

            # in such cases, it's hard to forecast for n > 1,
            # so sanitize!
            if n != 1:
                raise ValueError
        else:
            rdata = ForecastAlgorithms.rts(sub_series)
            fit = ForecastAlgorithms.rforecast.ets(rdata)
            forecast = ForecastAlgorithms.rforecast.forecast(fit, h=n)
            results = np.append(results, np.asarray(forecast[1]))

        return np.rint(results)


def main(args):
    # read csv
    series = pd.read_csv(
        args.data,  # processed dataset
        header=None,  # contains no header
        index_col=0,  # set datetime column as index
        names=['datetime', 'requests'],  # name the columns
        converters={'datetime':  # custom datetime parser
                        lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')},
        squeeze=True,  # convert to Series
        dtype={'requests': np.float64}  # https://git.io/vdbyk
    )

    # prediction at time 0 is NaN
    naive_results = np.array([np.nan])
    ar_results = np.array([np.nan])
    arma_results = np.array([np.nan])
    arima_results = np.array([np.nan])
    ets_results = np.array([np.nan])
    prev_observations = np.array([np.nan])
    current_observations = np.array([series[0]])

    # initialize forecasting algos
    algo = ForecastAlgorithms()

    # simulate forecast for each elapsed hour
    for hr in tqdm.tqdm(range(1, len(series) + 1)):
        hr_data = series[:hr]  # # hr = 1, 2, 3 ... len(series)
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
        # prev
        prev_observations = np.append(prev_observations,
                                      series[hr - 1])
        # curr
        if hr == len(series):
            # the last forecast doesn't have a corresponding true observation
            current_observations = np.append(current_observations,
                                             np.nan)
        else:
            current_observations = np.append(current_observations,
                                             series[hr])

    # replace < 0 with 0
    with warnings.catch_warnings():
        # ignore nan < 0 comparison warning
        warnings.filterwarnings('ignore')
        naive_results[naive_results < 0] = 0
        ar_results[ar_results < 0] = 0
        arma_results[arma_results < 0] = 0
        arima_results[arima_results < 0] = 0
        ets_results[ets_results < 0] = 0

    # save
    df_data = collections.OrderedDict()
    df_data['Naive'] = naive_results
    df_data['AR'] = ar_results
    df_data['ARMA'] = arma_results
    df_data['ARIMA'] = arima_results
    df_data['ETS'] = ets_results
    df_data['PreviousObservation'] = prev_observations
    df_data['CurrentObservation'] = current_observations
    pd.DataFrame(df_data, columns=df_data.keys()) \
        .to_csv(args.result_path, index=False, na_rep='NaN')


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='Path to processed dataset')
    parser.add_argument('result_path', type=str, help='Destination to save result')

    main(parser.parse_args())

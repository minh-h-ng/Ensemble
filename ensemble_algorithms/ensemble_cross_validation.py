# -*- coding: utf-8 -*-

import argparse
import collections
import logging
import sys

import numpy as np
import pandas as pd
from estimators import NaiveEstimator, ArEstimator, \
    ArmaEstimator, ArimaEstimator, EtsEstimator
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('data', type=str, help='Path to processed dataset')
parser.add_argument('result_path', type=str, help='Destination to save result')
args = parser.parse_args()

# Initialize logger
logger = logging.getLogger(__name__)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)


class EnsembleCrossValidation:
    def __init__(self, file_path, test_size=342):
        """
        Initializes data required by cross validation ensemble
        :param file_path: path to processed dataset
        :param test_size: last test_size hours will be treated as test set
        """
        # Read data frame
        self.series = pd.read_csv(
            file_path,
            header=None,  # contains no header
            index_col=0,  # set datetime column as index
            names=['datetime', 'requests'],  # name the columns
            converters={'datetime':  # custom datetime parser
                            lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')},
            squeeze=True,  # convert to Series
            dtype={'requests': np.float64},  # https://git.io/vdbyk
        )

        # Save test_size
        if test_size <= 0:
            raise ValueError
        self.test_size = test_size

        # exclude test data
        self.eval_series = self.series[:-test_size]
        self.test_series = self.series[-test_size:].copy()

        # be sure
        assert len(self.test_series) == test_size
        assert len(self.test_series) + len(self.eval_series) == len(self.series)

        logger.warning("Deciding best algorithm...")

        # Calculate mean score of 10-fold evaluation
        score = dict()
        logger.warning("Scoring Naive Algorithm")
        score['naive'] = cross_val_score(NaiveEstimator(), self.eval_series,
                                         cv=TimeSeriesSplit(n_splits=10).split(self.eval_series), verbose=3).mean()
        logger.warning("Scoring AR Algorithm")
        score['ar'] = cross_val_score(ArEstimator(), self.eval_series,
                                      cv=TimeSeriesSplit(n_splits=10).split(self.eval_series), verbose=3).mean()
        logger.warning("Scoring ARMA Algorithm")
        score['arma'] = cross_val_score(ArmaEstimator(), self.eval_series,
                                        cv=TimeSeriesSplit(n_splits=10).split(self.eval_series), verbose=3).mean()
        logger.warning("Scoring ARIMA Algorithm")
        score['arima'] = cross_val_score(ArimaEstimator(), self.eval_series,
                                         cv=TimeSeriesSplit(n_splits=10).split(self.eval_series), verbose=3).mean()
        logger.warning("Scoring ETS Algorithm")
        score['ets'] = cross_val_score(EtsEstimator(), self.eval_series,
                                       cv=TimeSeriesSplit(n_splits=10).split(self.eval_series), verbose=3).mean()

        # Find algo with min. score
        self.best_algo = min(score, key=score.get)

        logger.warning("Best Algorithm: %s" % self.best_algo)

    def run_test(self, result_path):
        # assign estimator
        if self.best_algo == 'naive':
            logger.warning("Running Naive Algorithm on Test Data")
            estimator = NaiveEstimator()
        elif self.best_algo == 'ar':
            logger.warning("Running AR Algorithm on Test Data")
            estimator = ArEstimator()
        elif self.best_algo == 'arma':
            logger.warning("Running ARMA Algorithm on Test Data")
            estimator = ArmaEstimator()
        elif self.best_algo == 'arima':
            logger.warning("Running ARIMA Algorithm on Test Data")
            estimator = ArimaEstimator()
        elif self.best_algo == 'ets':
            logger.warning("Running ETS Algorithm on Test Data")
            estimator = EtsEstimator()
        else:
            assert False

        # Makes eval series available for prediction
        estimator.fit(self.eval_series)

        # Run step-by-step prediction
        results = estimator.predict(self.test_series)

        df_data = collections.OrderedDict()
        df_data['Observation'] = self.test_series
        df_data['Prediction'] = results
        pd.DataFrame(df_data, columns=df_data.keys()) \
            .to_csv(result_path, index=False, na_rep='NaN')


def main():
    logger.warning("Starting Ensemble Cross Validation")

    # Initialize algo
    algo = EnsembleCrossValidation(file_path=args.data)

    # Run test
    algo.run_test(result_path=args.result_path)

    logger.warning("Stopping Ensemble Cross Validation")


if __name__ == '__main__':
    main()

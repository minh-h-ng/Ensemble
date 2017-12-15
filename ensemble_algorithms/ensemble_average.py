# -*- coding: utf-8 -*-

import argparse
import collections
import logging
import sys

import numpy as np
import pandas as pd
from estimators import AverageEstimator

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


class EnsembleAverage:
    def __init__(self, file_path, test_size=342):
        """
        Initializes data required by averaging ensemble
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

    def run_test(self, result_path):
        estimator = AverageEstimator()

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
    logger.warning("Starting Ensemble Average")

    # Initialize algo
    algo = EnsembleAverage(file_path=args.data, test_size=240)

    # Run test
    algo.run_test(result_path=args.result_path)

    logger.warning("Stopping Ensemble Average")


if __name__ == '__main__':
    main()

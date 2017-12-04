# -*- coding: utf-8 -*-

import argparse
import logging
import sys

import numpy as np
import pandas as pd

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('data', type=str, help='Path to dataset (output of forecast.py)')
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
        self.df = pd.read_csv(
            file_path,
            header=0,
            usecols=['Naive', 'AR', 'ARMA', 'ARIMA',
                     'ETS', 'CurrentObservation']
        )

        # Save test_size
        if test_size <= 0:
            raise ValueError
        self.test_size = test_size

    def run_test(self, result_path):
        # Copy test data
        dfs = self.df[-self.test_size:].copy()

        # Calculate average & ceil results
        dfs['Average'] = dfs[['Naive', 'AR', 'ARMA', 'ARIMA', 'ETS']].mean(axis=1).apply(np.ceil)

        # Save
        dfs.to_csv(result_path, columns=['Average'], header=False, index=False)


def main():
    logger.warning("Starting Ensemble Average")

    # Initialize algo
    algo = EnsembleAverage(file_path=args.data, test_size=342)

    # Run test
    algo.run_test(result_path=args.result_path)

    logger.warning("Stopping Ensemble Average")


if __name__ == '__main__':
    main()

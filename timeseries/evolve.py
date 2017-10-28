# -*- coding: utf-8 -*-

import json
import logging
import os
import sys
from functools import partial

import pandas as pd
from deap import base, creator, tools

# Initialize logger
logger = logging.getLogger(__name__)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

# single-objective maximization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Each individual is a list of time-series model weight
# [Naive, AR, ARMA, ARIMA, ETS]
creator.create("Individual", list, fitness=creator.FitnessMax)


# Initial population is seeded from json file
def initPopulation(pcls, ind_init, filename):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init(c) for c in contents)


def computeElasticityIndex(row):
    return min(row['GA'], row['CurrentObservation']) / max(row['GA'], row['CurrentObservation'])


def computeFitness(individual, hours_elapsed):
    # script's directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, '..', 'PythonESN', 'data', 'edgar')

    # WARN: the data must be sampled at every hour for nrows to work!

    # read frame
    df = pd.read_csv(file_path, nrows=hours_elapsed)

    # compute predicted value in hours_elapsed
    df['GA'] = df[['Naive', 'AR', 'ARMA', 'ARIMA', 'ETS']].dot(individual).round()

    # compute SEI for predictions in hours_elapsed
    return df.apply(computeElasticityIndex, axis=1).sum(),


def init_toolbox():
    toolbox = base.Toolbox()

    # Population is a list of individuals seeded from json file
    toolbox.register("population", initPopulation, list, creator.Individual, "seed_population.json")

    # fitness is computed using elasticity index
    toolbox.register("evaluate", computeFitness)

    # select best 50 individuals
    toolbox.register("select", tools.selBest, k=50)

    return toolbox


def main():
    toolbox = init_toolbox()

    population = toolbox.population()

    # Evaluate population
    fitnesses = list(map(partial(toolbox.evaluate, hours_elapsed=1), population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    fits = [ind.fitness.values[0] for ind in population]
    print(fits)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-

import json
import logging
import os
import random
import sys
from functools import partial

import numpy as np
import pandas as pd
from deap import base, creator, tools
from scoop import futures

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
    file_path = os.path.join(dir_path, '..', 'PythonESN', 'data_backup', 'edgar')

    # WARN: the data must be sampled at every hour for nrows to work!

    # read frame
    df = pd.read_csv(file_path, nrows=hours_elapsed)

    # compute predicted value in hours_elapsed
    if hours_elapsed >= 500:
        start_idx = hours_elapsed - (500 - 1)
        df['GA'] = df[['Naive', 'AR', 'ARMA', 'ARIMA', 'ETS']][start_idx:(500 + 1)] \
            .dot(individual).round()
    else:
        df['GA'] = df[['Naive', 'AR', 'ARMA', 'ARIMA', 'ETS']].dot(individual).round()

    # compute SEI for predictions in hours_elapsed
    return df.apply(computeElasticityIndex, axis=1).sum(),


def cxArithmeticAverage(ind1, ind2):
    _os = [sum(x) / 2 for x in zip(ind1, ind2)]
    return creator.Individual(_os)


def mutSwapGenes(individual):
    size = len(individual)

    rand = np.random.choice(range(size), 2, replace=False)
    indx1 = rand[0]
    indx2 = rand[1]

    individual[indx1], individual[indx2] = \
        individual[indx2], individual[indx1]

    return individual,


def init_toolbox():
    toolbox = base.Toolbox()

    # Population is a list of individuals seeded from json file
    toolbox.register("population", initPopulation, list, creator.Individual, "seed_population.json")

    # Evaluation: elasticity index
    toolbox.register("evaluate", computeFitness)

    # Selection: best 50 individuals
    toolbox.register("select", tools.selBest, k=50)

    # Crossover: Arithmetic average
    toolbox.register("mate", cxArithmeticAverage)

    # Mutation: swap genes
    toolbox.register("mutate", mutSwapGenes)

    # multiprocessing
    toolbox.register("map", futures.map)

    return toolbox


def varTwoByTwo(population, toolbox, cxpb, mutpb):
    offspring = []

    # "two-by-two" combinations
    for ind1_idx in range(len(population)):
        for ind2_idx in range(ind1_idx + 1, len(population)):
            if random.random() < cxpb:
                # clone
                ind1 = toolbox.clone(population[ind1_idx])
                ind2 = toolbox.clone(population[ind2_idx])

                # mate
                _os = toolbox.mate(ind1, ind2)
                del _os.fitness.values
                offspring.append(_os)

    # mutation
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def eaValter(population, toolbox, cxpb, mutpb, ngen, hours_elapsed,
             stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(partial(toolbox.evaluate, hours_elapsed=hours_elapsed), invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    # save the current best individual
    best_ind = toolbox.clone(halloffame[-1])

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        logger.debug(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varTwoByTwo(population, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(partial(toolbox.evaluate, hours_elapsed=hours_elapsed), invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # break if best_ind is not changed
        if best_ind == halloffame[-1]:
            break

        # Select the next generation population
        population[:] = toolbox.select(population + offspring)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            logger.debug(logbook.stream)

    return population, logbook


def run_ga(hours_elapsed):
    toolbox = init_toolbox()

    population = toolbox.population()

    # Parameters
    cxpb = 0.90  # Probability of mating two individuals
    mutpb = 0.10  # Probability of mutating an individual
    ngen = 100  # Number of generations

    # Best individual
    halloffame = tools.HallOfFame(maxsize=1)
    final_population, logbook = eaValter(population, toolbox, cxpb, mutpb, ngen,
                                         hours_elapsed=hours_elapsed, halloffame=halloffame)
    return halloffame[-1]


if __name__ == '__main__':
    # script's directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, '..', 'PythonESN', 'data_backup', 'edgar')

    for hours_elapsed in range(1, 2208):  # number of hours in edgar logs
        # read genes
        df = pd.read_csv(file_path, nrows=1,
                         skiprows=hours_elapsed,
                         header=None,
                         names=['Naive', 'AR', 'ARMA', 'ARIMA', 'ETS'],  # name the columns
                         usecols=[0, 1, 2, 3, 4]
                         )

        # evolve
        halloffame = run_ga(hours_elapsed)

        # print best
        series = df[['Naive', 'AR', 'ARMA', 'ARIMA', 'ETS']].dot(halloffame).round()
        assert series.size == 1
        print(series[0])

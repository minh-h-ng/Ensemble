# -*- coding: utf-8 -*-

import argparse
import json
import logging
import random
from functools import partial

import numpy as np
import pandas as pd
import tqdm
from deap import base, creator, tools
from scoop import futures

# Initialize logger
logger = logging.getLogger(__name__)
sh = logging.FileHandler('evolve.log', 'w')
sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

# single-objective maximization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Each individual is a list of time-series model weight
# [Naive, AR, ARMA, ARIMA, ETS]
creator.create("Individual", list, fitness=creator.FitnessMax)


class GeneticAlgorithm:
    def __init__(self, file_path, samples=500,
                 cxpb=0.90, mutpb=0.10, ngen=100,
                 selbest=50):

        """Initializes data required by genetic algorithm
        :param file_path: path to output of forecast.py
        :param samples: clip algorithm to most recent 500 forecasts
        :param cxpb: probability of mating two individuals
        :param mutpb: probability of mutating an individual
        :param ngen: number of generations
        :param selbest: number of best individuals to select
        """
        # Read data frame
        self.df = pd.read_csv(
            file_path,
            header=0,
            usecols=['Naive', 'AR', 'ARMA', 'ARIMA',
                     'ETS', 'CurrentObservation']
        )

        # Clip
        if samples <= 0:
            raise ValueError
        self.clip = samples

        # Parameters
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.selbest = selbest

    # Initial population is seeded from json file
    def initPopulation(self, pcls, ind_init, filename):
        with open(filename, "r") as pop_file:
            contents = json.load(pop_file)
        return pcls(ind_init(c) for c in contents)

    def computeElasticityIndex(self, row):
        return min(row['GA'], row['CurrentObservation']) / max(row['GA'], row['CurrentObservation'])

    def computeFitness(self, individual, hours_elapsed):
        # df subset
        if hours_elapsed > self.clip:
            start_idx = hours_elapsed - self.clip  # # (1, 2, 3, 4 ...)
            end_idx = hours_elapsed  # # (501, 502, 503, 504 ...)
            dfs = self.df[start_idx:end_idx].copy()
        else:
            end_idx = hours_elapsed  # # (0:1, 0:2, 0:3, 0:4 ... 0:500)
            dfs = self.df[:end_idx].copy()

        # compute predicted value in hours_elapsed
        dfs['GA'] = dfs[['Naive', 'AR', 'ARMA', 'ARIMA', 'ETS']].dot(individual).round()

        # compute SEI for predictions in hours_elapsed
        return dfs.apply(self.computeElasticityIndex, axis=1).sum(),

    def cxArithmeticAverage(self, ind1, ind2):
        _os = [sum(x) / 2 for x in zip(ind1, ind2)]
        return creator.Individual(_os)

    def mutSwapGenes(self, individual):
        size = len(individual)

        rand = np.random.choice(range(size), 2, replace=False)
        indx1 = rand[0]
        indx2 = rand[1]

        individual[indx1], individual[indx2] = \
            individual[indx2], individual[indx1]

        return individual,

    def varTwoByTwo(self, population, toolbox, cxpb, mutpb):
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

    def eaValter(self, population, toolbox, cxpb, mutpb, ngen, hours_elapsed,
                 stats=None, halloffame=None):
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
        logger.warning('\n' + logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Vary the population
            offspring = self.varTwoByTwo(population, toolbox, cxpb, mutpb)

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
                logger.warning("Best Individual didn't change. Stopping evolution")
                break
            else:
                best_ind = toolbox.clone(halloffame[-1])

            # Select the next generation population
            population[:] = toolbox.select(population + offspring)

            # Update the statistics with the new population
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            logger.warning('\n' + logbook.stream)

        return population, logbook

    def init_toolbox(self):
        toolbox = base.Toolbox()

        # Population is a list of individuals seeded from json file
        toolbox.register("population", self.initPopulation,
                         list, creator.Individual, "seed_population.json")

        # Evaluation: elasticity index
        toolbox.register("evaluate", self.computeFitness)

        # Selection: best 50 individuals
        toolbox.register("select", tools.selBest, k=self.selbest)

        # Crossover: Arithmetic average
        toolbox.register("mate", self.cxArithmeticAverage)

        # Mutation: swap genes
        toolbox.register("mutate", self.mutSwapGenes)

        # multiprocessing
        toolbox.register("map", futures.map)

        return toolbox

    def run(self, hours_elapsed):
        toolbox = self.init_toolbox()

        population = toolbox.population()

        # Best individual
        halloffame = tools.HallOfFame(maxsize=1)
        final_population, logbook = self.eaValter(population, toolbox,
                                                  self.cxpb, self.mutpb, self.ngen,
                                                  hours_elapsed=hours_elapsed,
                                                  halloffame=halloffame)

        logger.warning("Hall of Fame: " + str(halloffame[-1]))

        # read genes at hours_elapsed
        # iloc is zero-indexed
        genes = self.df.iloc[hours_elapsed - 1][['Naive', 'AR', 'ARMA', 'ARIMA', 'ETS']]

        # compute GA estimate
        ga_estimate = genes.dot(halloffame[-1]).round()

        logger.warning("GA Estimate: " + str(ga_estimate))
        return ga_estimate


def main(args):
    logger.warning("Starting GA")

    # sanitize (elapsed hours should at-least be 1)
    if args.hours_start <= 0:
        logger.critical("Invalid args. Aborting!")
        return

    # initialize
    ga = GeneticAlgorithm(args.data)

    # run for every hour
    for hr in tqdm.tqdm(range(args.hours_start, args.hours_end + 1)):
        logger.warning("Processing hour: " + str(hr))
        result = ga.run(hr)
        with open(args.result_path + '_' + args.times, 'a') as f:
            f.write(str(result) + '\n')
        logger.warning('\n')

    logger.warning("Stopping GA")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('times', type=str, help='Times being executed. It is appended to results file')
    parser.add_argument('data', type=str, help='Path to dataset (output of forecast.py)')
    parser.add_argument('hours_start', type=int, help='Hour to begin processing from (inclusive)')
    parser.add_argument('hours_end', type=int, help='Hour to end processing at (inclusive)')
    parser.add_argument('result_path', type=str, help='Destination to save result')

    main(parser.parse_args())
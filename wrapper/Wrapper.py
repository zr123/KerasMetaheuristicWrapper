from random import randint, uniform, random

import keras.backend as K
import numpy as np
from functools import reduce


class Wrapper:
    def __init__(self, model, x, y):
        self.model = model
        self.x = x
        self.y = y
        self.weight_dimensions = [w.shape for w in self.model.get_weights()]
        # self.particles = [self.create_model() for p in range(population)]

    def get_weights_from_chromosome(self, chromosome):
        # change pyswarms-chromsome (1-dimensional numpy array) to keras weights (array of numpy arrays)
        offset = 0
        weights = []
        for w_dim in self.weight_dimensions:
            weight_count = reduce(lambda x, y: x * y, w_dim)
            weights.append(chromosome[offset: offset + weight_count].reshape(w_dim))
            offset += weight_count
        return weights

    def get_dimensions(self):
        # calculated the number of trianable weights
        return int(np.sum([K.count_params(p) for p in set(self.model.trainable_weights)]))

    def get_chromosome(self):
        weights = np.empty(0)
        for w in self.model.get_weights():
            weights = np.append(weights, w.flatten())
        return weights

    def get_fitness(self, chromosome):
        # calculates the fitness of the model for one chromosome
        # set weights
        self.model.set_weights(self.get_weights_from_chromosome(chromosome))
        # return the loss - calculate on a single batch to get the stochastic loss
        return self.model.evaluate(self.x, self.y, verbose=0, steps=1)[0]

    def evaluate(self, chromosome, x, y):
        self.model.set_weights(self.get_weights_from_chromosome(chromosome))
        return self.model.evaluate(x, y, verbose=0)

    def get_all_fitnesses(self, chromosomes):
        n_particles = chromosomes.shape[0]
        j = [self.get_fitness(chromosomes[i]) for i in range(n_particles)]
        return np.array(j)

    def create_random_chromosome(self):
        return np.random.uniform(-1.0, 1.0, self.get_dimensions())

    def my_fitenss_as_tuple(self, individual):
        # deap expects fitness function to return a tuple
        return self.get_fitness(individual),

    def crossover(self, chromosome1, chromosome2):
        size = len(chromosome1)
        cxpoint1 = randint(1, size)
        cxpoint2 = randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        chromosome1[cxpoint1:cxpoint2], chromosome2[cxpoint1:cxpoint2] \
            = chromosome2[cxpoint1:cxpoint2].copy(), chromosome1[cxpoint1:cxpoint2].copy()

        return chromosome1, chromosome2

    def mutate(self, chromosome):
        # randomly assign a single weight or bias a random number [-1.0, 1.0]
        location = randint(0, chromosome.shape[0] - 1)
        chromosome[location] = uniform(-1.0, 1.0)
        return chromosome


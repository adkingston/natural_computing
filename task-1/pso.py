"""
Implementation of Particle Swarm Optimizer.

This implementation will be sequential to allow new information to be utilized
as soon as it becomes available
"""

import random as rd
import numpy as np
import torch

# start with particle


def init_pos(limit):
    """ limit is a 2 element list where elt 1 < elt 2 """
    return limit[0] + (limit[1] - limit[0]) * rd.random()


class Particle:
    """
    Particle used in particle swarm
    :param dimension: integer
    """

    def __init__(self, dimension, limit):
        """ limit is a 2 element list """
        self.dimension = dimension
        self.limit = limit
        self.x = torch.cuda.FloatTensor(
            [init_pos(limit) for x in range(dimension)])
        self.v = torch.cuda.FloatTensor(
            [0.1 for x in range(dimension)])
        self.best = self.x.data
        self.best_fitness = np.inf

    def update(self, omega, alpha_1, alpha_2, g):
        """ update the velocity and position of the particle """
        rand_1 = torch.cuda.FloatTensor(
            [rd.random() for x in range(self.dimension)])
        rand_2 = torch.cuda.FloatTensor(
            [rd.random() for x in range(self.dimension)])

        self.v = (omega * self.v + (alpha_1 * rand_1) * (
            self.best - self.x) + (alpha_2 * rand_2) * (g - self.x)).data

        self.x = (self.x + self.v).data

    def update_best(self, objective):
        """ if the current position is better than the current best, update """
        curr_fitness = objective(self.x)
        if curr_fitness < self.best_fitness:
            self.best = self.x
            self.best_fitness = curr_fitness


class Swarm:
    """
    Swarm used to optimize the objective function
    :param num: integer - the size of the swarm
    :param dimension: integer - the dimension of the particel
    :param objective_function: function(torch.cuda.FloatTensor) -> float - the function to
    minimize
    """

    def __init__(
            self,
            num,
            dimension,
            limit,
            omega,
            alpha_1,
            alpha_2):
        """ initialize the swarm with the necessary components """
        self.omega = omega
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.g = torch.cuda.FloatTensor(
            [0 for x in range(dimension)])
        self.swarm = [Particle(dimension, limit) for x in range(num)]
        self.best_global_fitness = np.inf

    def perform_iteration(self, objective):
        """ perform a single iteration on the entire swarm """
        for particle in self.swarm:
            # update the particle's velocity and position
            particle.update(
                self.omega,
                self.alpha_1,
                self.alpha_2,
                self.g)

            # check if the particle's best position needs to be updated
            particle.update_best(objective)

            # check if the particle's new pest is better than the current
            # global best
            if particle.best_fitness < self.best_global_fitness:
                self.g = particle.x.data
                self.best_global_fitness = particle.best_fitness

    def optimize(self, objective_function, num_iterations=200):
        """ perform the particle swarm algorithm.  """
        for _ in range(num_iterations):
            self.perform_iteration(objective_function)

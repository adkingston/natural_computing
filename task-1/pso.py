"""
Implementation of Particle Swarm Optimizer.

This implementation will be sequential to allow new information to be utilized
as soon as it becomes available
"""

import numpy as np
import math

# start with particle


class Particle:
    """
    Particle used in particle swarm
    :param dimension: integer
    """

    def __init__(self, dimension):
        """ 3 is chosen arbitrarily """
        self.dimension = dimension
        self.particle_position = np.random.uniform(
            low=-3.0, high=3.0, size=(1, dimension))
        self.particle_velocity = np.random.uniform(
            low=-3.0, high=3.0, size=(1, dimension))
        self.best = np.array([3.0 for x in range(dimension)], dtype=np.float64)

    def update(self, omega, alpha_1, alpha_2, global_best):
        """ update the velocity and position of the particle """
        rand_1 = np.random.uniform(low=0.0, high=1.0, size=(1, self.dimension))
        rand_2 = np.random.uniform(low=0.0, high=1.0, size=(1, self.dimension))

        self.particle_velocity = omega * self.particle_velocity + (alpha_1 * rand_1) * (
            self.best - self.particle_position) + (alpha_2 * rand_2) * (global_best - self.particle_position)

        self.particle_position += self.particle_velocity

    def update_best(self, objective):
        """ if the current position is better than the current best, update """
        if objective(self.particle_position) < objective(self.best):
            self.best = self.particle_position


class Swarm:
    """
    Swarm used to optimize the objective function
    :param num: integer - the size of the swarm
    :param dimension: integer - the dimension of the particel
    :param objective_function: function(np.array) -> float - the function to
    minimize
    """

    def __init__(
            self,
            num,
            dimension,
            omega,
            alpha_1,
            alpha_2):
        """ initialize the swarm with the necessary components """
        self.omega = omega
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        big_effin_number = 1e5  # not huge, but big enough
        self.global_best = np.array(
            [big_effin_number for x in range(dimension)])
        self.swarm = [Particle(dimension) for x in range(num)]

    def perform_iteration(self, objective):
        """ perform a single iteration on the entire swarm """
        for particle in self.swarm:
            # update the particle's velocity and position
            particle.update(
                self.omega,
                self.alpha_1,
                self.alpha_2,
                self.global_best)

            # check if the particle's best position needs to be updated
            particle.update_best(objective)

            # check if the particle's new pest is better than the current
            # global best
            if objective(
                    particle.best) < objective(
                    self.global_best):
                self.global_best = particle.best

        print(self.global_best)

    def optimize(self, objective_function, num_iterations=200):
        """ perform the particle swarm algorithm """
        for _ in range(num_iterations):
            self.perform_iteration(objective_function)


def two_dimensional_parabola(vector):
    """ expects np array with 2 elements """
    def f(x): return np.abs(2 - x)**2
    return np.sum(np.array([f(x) for x in vector]))


swarm = Swarm(num=20, dimension=2, omega=0.5, alpha_1=1.0, alpha_2=1.0)

swarm.optimize(two_dimensional_parabola, num_iterations=75)

print(swarm.global_best)

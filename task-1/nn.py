"""
Implementation of the Neural Network
"""

import numpy as np
import pso


def two_dimensional_parabola(vector):
    """ expects np array with 2 elements """
    def f(x): return np.abs(2 - x)**2
    return np.sum(np.array([f(x) for x in vector]))


SWARM = pso.Swarm(num=20, dimension=2, omega=0.5, alpha_1=1.0, alpha_2=1.0)

SWARM.optimize(two_dimensional_parabola, num_iterations=75)

print(SWARM.global_best)

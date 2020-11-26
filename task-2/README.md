# Task 2 

In this directory there is python file called ga_nn_evolution.py. This file can be run using the command:

python ga_nn_evolution.py

This will run the genetic algorithm and put the results in GA_output.txt.

The dependencies of the above program are specified below:
itertools
torch
numpy
json
from gen_algorithm import geneticalgorithm from gen_algorithm.py in the included folder.

You can change the GA parameters using the algorithm_param variable. An example is provided below
algorithm_param = {'max_num_iteration': 100,
                   'population_size':30,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None}
                   

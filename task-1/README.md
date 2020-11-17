# Task 1 

In this directory there is an executible called `cross-validate`. This program will run a K-Fold Cross Validation on 4 possible neural networks:
1. A PSO optimised neural net with non-linear inputs (shape=(6,8,1))
2. A PSO optimised neural net with linear inputs (shape=(2,6,5,1))
3. A SGD optimised neural net with non-linear inputs (shape=(6,8,1))
4. A SGD optimised neural net with linear inputs (shape=(2,6,5,1))

The benefit of this executible is that each validation is run concurrently (see the Go code in the `cross_validator` directory).

To see the possible inputs, run `./cross-validate --help`. 

There are 3 parameters:
`-folds [int]` the number of folds for cross validation
`-optimiser [pso/sgd]` which optimiser to use when training
`-inputType [linear/nonlinear]` what type of inputs to use 

Running just `./cross-validate` will resuls in all four possible neural nets being trained on 10-fold cross validation.

To generate graphs, run `python plot_pso_loss.py`. 

In `output.json` you will find the results from each experiment. As an example, below is the output of running `./cross-validate`:
```
[
  {
    "iterations": 10,
    "optimizer": "pso",
    "input_type": "linear",
    "average_training_loss": 0.06567051799324916,
    "average_testing_loss": 0.12478226452871999,
    "average_fitness": 0.15742747857599376,
    "best_iteration": 9
  },
  {
    "iterations": 10,
    "optimizer": "pso",
    "input_type": "nonlinear",
    "average_training_loss": 0.015866880337198033,
    "average_testing_loss": 0.03782255463314025,
    "average_fitness": 0.048872953739163824,
    "best_iteration": 9
  },
  {
    "iterations": 10,
    "optimizer": "sgd",
    "input_type": "linear",
    "average_training_loss": 0.16390071632340547,
    "average_testing_loss": 0.26545349185665446,
    "average_fitness": 0.31622987962327903,
    "best_iteration": 9
  },
  {
    "iterations": 10,
    "optimizer": "sgd",
    "input_type": "nonlinear",
    "average_training_loss": 0.08283769022673368,
    "average_testing_loss": 0.11639524325728416,
    "average_fitness": 0.1331740197725594,
    "best_iteration": 9
  }
]
```

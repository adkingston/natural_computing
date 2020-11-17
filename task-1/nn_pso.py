"""
An experiment designed to break down the neural network to its most basic
form - composition of functions on matrices
"""

# note that functions on floats can be applied elementwise to tensors

import sys
import json
import numpy as np
import pso
import nn_data as data


def count_layer_weights(ab, count_bias=True):
    """
    counts the edges + biases between two layers in a neural network
    modifier is used if biases are not there
    """
    if count_bias:
        return ab[1] * (ab[0] + 1)
    return ab[1] * ab[0]


def get_layer_weight_shape(ab, with_bias=True):
    """
    gets the shape of the weight vector accounting for biases
    (can be turned off)
    """
    if with_bias:
        return [ab[1], ab[0] + 1]
    return ab[::-1]  # reverse order


def get_weight_split(shape=(2, 1)):
    """
    given the shape of a network, this will return the index range of the
    weights for each layers calculation, and the shape that the subvector would
    need to be transformed into.

    e.g: if we are calculating between the first and second layers (with bias),
    with 2 and 4 nodes respectively, then we would want the first 12 elements
    of the weight vector transformed into a 3x4 matrix
    """
    nds = [[0, 0]] + [[shape[i], shape[i + 1]] for i in range(len(shape) - 1)]
    split = []
    for j in range(1, len(nds)):
        if j == 1:
            split.append([[0, count_layer_weights(nds[j])],
                          get_layer_weight_shape(nds[j])])
        else:
            split.append([[split[j - 2][0][1],
                           split[j - 2][0][1] + count_layer_weights(nds[j])],
                          get_layer_weight_shape(nds[j],
                                                 (j != len(nds) - 1))])
    return split


def make_neural_network(shape=(2, 1), activation_function=np.sin):
    """
    Takes in the shape of the neural network, and returns a closure that
    accepts a weight tensor as its parameter. This tensor is used as the weights
    and biases for the neural network
    """
    # want a list in the following format:
    # [[indices of weight vector, shape of weight matrix]]
    # this will tell us which part of the weight vector, and the shape it has
    # to take at each level.
    split = get_weight_split(shape)

    def nn(weights, inp):
        for sp in split:
            if sp != split[-1]:
                inp = np.append(inp, [1])
            index, sh = sp[0], sp[1]
            W = np.asarray(weights[index[0]: index[1]]).reshape(sh)

            inp = activation_function(W @ inp)

        return inp
    return nn


def get_spiral_data(filename):
    """ gets data set from file """
    out = []
    with open(filename) as data:
        for line in data:
            x1, x2, y = line.split()
            vec = np.array([float(x1), float(x2), float(
                x1)**2, float(x2)**2, np.sin(float(x1)), np.sin(float(x2))])
            y = np.array([float(y)])
            out.append([vec, y])

    return out


def get_cost_fn(nn, inp, actual):
    def cost(weights):
        pred = np.array([nn(weights, x) for x in inp])
        return float(sum(np.abs(actual - pred)**2) / len(actual))
    return cost


def pso_nonlinear():
    """
    Train and test a neural network
    """
    # get data
    filename = sys.argv[1]
    train_raw = data.get_spiral_data(filename + "/train.dat", data.T6)
    test_raw = data.get_spiral_data(filename + "/test.dat", data.T6)

    x_train = np.array([x[0] for x in train_raw])
    y_train = np.array([x[1] for x in train_raw])

    x_test = np.array([x[0] for x in test_raw])
    y_test = np.array([x[1] for x in test_raw])

    # define neural network
    shape = (6, 8, 1)
    net = make_neural_network(shape)
    out = train_nn(net, shape, x_train, y_train, x_test, y_test)
    print(json.dumps(out))


def pso_linear():
    """
    Train and test a neural network
    """
    # get data
    filename = sys.argv[1]
    train_raw = data.get_spiral_data(filename + "/train.dat", data.T2)
    test_raw = data.get_spiral_data(filename + "/test.dat", data.T2)

    x_train = np.array([x[0] for x in train_raw])
    y_train = np.array([x[1] for x in train_raw])

    x_test = np.array([x[0] for x in test_raw])
    y_test = np.array([x[1] for x in test_raw])

    # define neural network
    shape = (2, 8, 5, 1)
    net = make_neural_network(shape)
    out = train_nn(net, shape, x_train, y_train, x_test, y_test)
    print(json.dumps(out))


def train_nn(net, shape, x_train, y_train, x_test, y_test):
    cost = get_cost_fn(net, x_train, y_train)

    swarm = pso.Swarm(
        num=10,
        dimension=data.get_dimension(shape),
        limit=[-0.5, 0.5],
        omega=0.7,
        alpha_1=1.5,
        alpha_2=1.5
    )

    epochs = 5000

    test_cost = get_cost_fn(net, x_test, y_test)
    swarm.optimize(cost, test_cost, epochs)

    e_train = swarm.best_global_fitness
    e_test = np.float64(swarm.test_loss_history[-1])
    fitness = 1 - ((e_train + e_test) / 2. + e_test - e_train)
    return {
        'training_loss': swarm.best_global_fitness,
        'train_epochs': swarm.train_loss_history,
        'testing_loss': np.float64(swarm.test_loss_history[-1]),
        'fitness': fitness,
        'test_epochs': swarm.test_loss_history,
    }


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == 'linear':
        pso_linear()
    else:
        pso_nonlinear()

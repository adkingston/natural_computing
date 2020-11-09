"""
Implementation of the Neural Network
"""

import numpy as np
from torch import nn
from collections import OrderedDict
import numpy as np
import torch
import pso


def new_net(shape, activator=nn.ReLU):
    """
    Creates a new neural network
    Assumes the same activiation function and linear layer type between each
    level
    Params:
    shape [INT]: each element is the number of nodes for that layer
        e.g. [2,1] denotes a NN with two input nodes, one output, and no
        hidden layers (a perceptron)

    activator callable: the torch.nn activator function. defaults to ReLU

    """
    if shape is None or len(shape) == 0:
        return None

    layers = []
    for index in range(0, len(shape) - 1):
        layer_name = "fc" + str(index + 1)
        activation_name = (activator.__name__ + str(index + 1)).lower()

        layers.append((layer_name, nn.Linear(shape[index], shape[index + 1])))
        layers.append((activation_name, activator()))

    layers.append(('final', nn.Sigmoid()))

    return nn.Sequential(OrderedDict(layers))


def get_dimension(shape):
    """
    given the shape of a neural network, calculate the dimension of
    the weight vector
    Params:
    shape [INT]: each element is the number of nodes in a layer of a net
    """
    mean_squared = 0
    for index in range(len(shape) - 1):
        mean_squared += shape[index + 1] * (shape[index] + 1)

    return mean_squared + 1


def update_weights(net, weights):
    """
    given a neural network and a weight vector, update the weights of the net
    Params:
    net nn.Module: A neural network
    weights [float64]: the weights for the neural network
    """
    with torch.no_grad():
        running_total = 0
        # set weights
        for params in net.parameters():
            params.copy_(torch.reshape(torch.FloatTensor(
                weights[running_total:
                        np.prod(params.size()) + running_total]),
                params.size()))

            running_total += np.prod(params.size())


def squared_error(pred, val):
    """
    Squared error calculates the mean squeared error between two vectors
    """
    pred = pred.detach().numpy()
    return sum(np.abs(val - pred)**2) / len(val)


def fitness(point, actual_value, net, cost_func):
    """
    calculates the fitness of a neural network for a given cost function
    """
    predicted_value = net(point)
    cost = cost_func(predicted_value, actual_value)
    return cost.item()


def objective(point, actual_value, net, loss_func):
    """
    A closure that takes in a batch of training/testing points, their actual
    values, a neural network and a cost function.

    Returns: a function that takes a weight vector as input and returns the
    value of the cost function
    """
    def retfunc(weights):
        update_weights(net, weights)
        return fitness(point, actual_value, net, loss_func)

    return retfunc

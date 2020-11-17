"""
implements optimizer interface by pytorch
"""

import sys
import json
from collections import OrderedDict
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import nn_data as data


class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)


def new_net(shape, activator=nn.ReLU):
    """
    Creates a new neural network
    Assumes the same activiation function and linear layer type between each
    level
    Params:
    shape (INT): each element is the number of nodes for that layer
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


def train_nn(net, train_loader, test_loader):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.03, momentum=0.4)

    train_history = []
    test_history = []

    for _ in range(5000):
        training_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        testing_loss = 0.0
        for inputs, labels in test_loader:
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            testing_loss += loss.item()

        training_loss /= len(train_loader)
        testing_loss /= len(test_loader)

        train_history.append(str(training_loss))
        test_history.append(str(testing_loss))

    return {
        'training_loss': np.float64(train_history[-1]),
        'testing_loss': np.float64(test_history[-1]),
        'train_epochs': train_history,
        'test_epochs': test_history
    }


def sgd_linear():
    filename = sys.argv[1]
    train_raw = data.get_spiral_data(filename + "/train.dat", data.T2, True)
    test_raw = data.get_spiral_data(filename + "/test.dat", data.T2, True)

    train, test = data.SpiralDataset(train_raw), data.SpiralDataset(test_raw)

    train_loader = DataLoader(train, 10)
    test_loader = DataLoader(test, 10)

    net = new_net((2, 8, 5, 1), nn.Tanh)

    out = train_nn(net, train_loader, test_loader)
    print(json.dumps(out))


def sgd_non_linear():
    filename = sys.argv[1]
    train_raw = data.get_spiral_data(filename + "/train.dat", data.T6, True)
    test_raw = data.get_spiral_data(filename + "/test.dat", data.T6, True)

    train, test = data.SpiralDataset(train_raw), data.SpiralDataset(test_raw)

    train_loader = DataLoader(train, 10)
    test_loader = DataLoader(test, 10)

    net = new_net((6, 8, 1), Sin)

    out = train_nn(net, train_loader, test_loader)
    print(json.dumps(out))


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == 'linear':
        sgd_linear()
    else:
        sgd_non_linear()

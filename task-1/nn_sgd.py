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
import matplotlib.pyplot as plt


class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)


class NeuralNetwork(nn.Module):
    '''
    Args:
        nn_list = [input dim., first hidden layer size,...,last hidden layer size, output dim.]

    '''

    def __init__(self, nn_list):
        super(NeuralNetwork, self).__init__()
        self.nn_list = nn_list
        self.lin_layers = nn.ModuleList()
        self.add_layers()

    def forward(self, x):
        # forward pass through the network
        for layer in self.lin_layers:
            x = torch.tanh(layer(x))

        # the last layer is a sigmoid activation layer
        # x = nn.functional.sigmoid(self.lin_layers[-1](x))

        return x

    def add_layers(self):
        # adjust nn_list
        length = len(self.nn_list)

        # adjust lin_layers
        for i in range(length - 1):
            self.lin_layers.append(
                nn.Linear(self.nn_list[i], self.nn_list[i + 1]))

# def new_net(shape, activator=nn.ReLU):
    # """
    # Creates a new neural network
    # Assumes the same activiation function and linear layer type between each
    # level
    # Params:
    # shape (INT): each element is the number of nodes for that layer
        # e.g. [2,1] denotes a NN with two input nodes, one output, and no
        # hidden layers (a perceptron)

    # activator callable: the torch.nn activator function. defaults to ReLU

    # """
    # if shape is None or len(shape) == 0:
        # return None

    # layers = []
    # for index in range(0, len(shape) - 1):
        # layer_name = "fc" + str(index + 1)
        # activation_name = (activator.__name__ + str(index + 1)).lower()

        # layers.append((layer_name, nn.Linear(shape[index], shape[index + 1])))
        # layers.append((activation_name, activator()))

    # # layers.append(('final', nn.Sigmoid()))

    # return nn.Sequential(OrderedDict(layers))


def train_nn(nnet, train_loader, test_loader):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    nnet.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(nnet.parameters(), lr=0.1)

    train_history = []
    test_history = []

    for ep in range(5000):
        training_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            nnet.train()
            outputs = nnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            del inputs
            del labels

        testing_loss = 0.0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = nnet(inputs)
            loss = criterion(outputs, labels)

            testing_loss += loss.item()

            del inputs
            del labels

        training_loss /= len(train_loader)
        testing_loss /= len(test_loader)

        train_history.append(str(training_loss))
        test_history.append(str(testing_loss))

        # print(
        # f"epoch: {ep}, training loss: {training_loss},
        # testing_loss:{testing_loss}")

    train_loss = np.float64(train_history[-1])
    test_loss = np.float64(test_history[-1])
    fitness = 1 - ((train_loss + test_loss) / 2.0 + test_loss - train_loss)
    return {
        'training_loss': train_loss,
        'testing_loss': test_loss,
        'fitness': fitness,
        'train_epochs': train_history,
        'test_epochs': test_history
    }


def sgd_linear():
    filename = sys.argv[1]
    train_raw = data.get_spiral_data(filename + "/train.dat", data.T2, True)
    test_raw = data.get_spiral_data(filename + "/test.dat", data.T2, True)

    train, test = data.SpiralDataset(train_raw), data.SpiralDataset(test_raw)

    train_l = DataLoader(train, 200)
    test_l = DataLoader(test, 200)

    net = NeuralNetwork((2, 8, 5, 1))

    out = train_nn(net, train_l, test_l)
    print(json.dumps(out))


def sgd_non_linear():
    filename = sys.argv[1]
    train_raw = data.get_spiral_data(filename + "/train.dat", data.T6, True)
    test_raw = data.get_spiral_data(filename + "/test.dat", data.T6, True)

    train, test = data.SpiralDataset(train_raw), data.SpiralDataset(test_raw)

    train_l = DataLoader(train, 200)
    test_l = DataLoader(test, 200)

    net = NeuralNetwork((4, 8, 1))

    out = train_nn(net, train_l, test_l)
    print(json.dumps(out))


if __name__ == '__main__':
    # full = data.get_spiral_data('two_spirals.dat', data.T6, True)
    # train = full[:int(len(full) / 2.)]
    # test = full[int(len(full) / 2.)::]

    # train_d, test_d = data.SpiralDataset(
    # train), data.SpiralDataset(test)

    # train_l = DataLoader(train_d, batch_size=10)
    # test_l = DataLoader(test_d, batch_size=10)

    # shape = (4, 8, 1)
    # net = NeuralNetwork(shape)
    # out = train_nn(net, train_l, test_l)
    # print(json.dumps(out))

    # x_span = np.linspace(-5, 5, 1000)
    # y_span = np.linspace(-5, 5, 1000)

    # xx, yy = np.meshgrid(x_span, y_span)

    # labels = net(torch.FloatTensor(np.c_[[[x, y, np.sin(x), np.sin(
    # y)] for x, y in np.c_[xx.ravel(), yy.ravel()]]])).detach().numpy()

    # z = labels.reshape(xx.shape)

    # plt.figure()
    # plt.contourf(xx, yy, z, cmap='RdBu', alpha=0.5)

    # X = [[x, y] for x in np.arange(-6.0, 6.0, 0.1)
    # for y in np.arange(-6.0, 6.0, 0.1)]

    # Y = [[x, y, 'go'] if net(torch.FloatTensor(
    # [x, y, np.sin(x), np.sin(y)])) < 0.5 else [x, y, 'bo'] for x, y in X]

    # plt.figure()
    # for point in Y:
    # plt.plot(point[0], point[1], point[2], markeredgewidth=0.0)

    # for point in train:
    # if point[1] == 0:
    # color = 'darkred'
    # else:
    # color = 'navy'
    # plt.plot(point[0][0], point[0][1], color=color, marker='o')

    # for point in test:
    # if point[1] == 0:
    # color = 'orangered'
    # else:
    # color = 'royalblue'
    # plt.plot(point[0][0], point[0][1], color=color, marker='o')
    # plt.show()

    # for point in test:
    # plt.plot(point[0][0], point[0][1], 'yo')
    # plt.show()

    if len(sys.argv) > 2 and sys.argv[2] == 'linear':
        sgd_linear()
    else:
        sgd_non_linear()

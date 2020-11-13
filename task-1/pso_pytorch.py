"""
implements optimizer interface by pytorch
"""

from torch import nn
import torch
import time
import pso
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim
import matplotlib.pyplot as plt
import nn as my_net
import random as rd
import sys
import json


class SpiralDataset(Dataset):
    """ implements pytorch dataset class """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]


def get_spiral_data(filename):
    """ gets data set from file """
    out = []
    with open(filename) as data:
        for line in data:
            x1, x2, y = line.split()
            vec = torch.FloatTensor([float(x1), float(x2), float(
                x1)**2, float(x2)**2, np.sin(float(x1)), np.sin(float(x2))])
            y = torch.FloatTensor([float(y)])
            out.append([vec, y])

    return out


def partition(lst, n=2):
    """
    partitions list into n equal parts
    (last partition may not be equal in size)
    """
    batch_size = int(len(lst) / n)
    retval = [lst[batch_size * i:batch_size * (i + 1) - 1] for i in range(n)]
    return retval


def get_train_test_sets(partitioned_dataset, iteration):
    """ take partitioned dataset and return the full test and training sets """
    training_set = [partitioned_dataset[i][x] for i in range(len(
        partitioned_dataset)) if i != iteration for x in range(len(partitioned_dataset[i]))]

    return training_set, partitioned_dataset[iteration]


def time_it(func):
    """ gives run time of a function """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        loss = func(*args, **kwargs)
        print(f'{func.__name__} took {time.time()-start_time} seconds')
        return loss
    return wrapper


def train(net, point, value, swarm):
    DEVICE = torch.device("cuda:0")
    point, value = point.to(DEVICE), value.to(DEVICE)
    net.train()

    objective_func = my_net.objective(
        point, value, net, my_net.squared_error)
    swarm.perform_iteration(objective_func)

    my_net.update_weights(net, swarm.g)

    prediction = net(point)
    loss = my_net.squared_error(prediction, value)
    return loss


def test(net, point, value):
    DEVICE = torch.device("cuda:0")
    point, value = point.to(DEVICE), value.to(DEVICE)
    prediction = net(point)
    loss = my_net.squared_error(prediction, value)
    return loss


def train_and_test(
        train_loader,
        test_loader,
        epochs,
        shape,
        omega,
        alpha1,
        alpha2,
        print_epoch=False):
    """ trains and test a nn given shape and swarm parameters """

    DEVICE = torch.device("cuda:0")
    # setup
    neural_net = my_net.new_net(shape=shape, activator=my_net.Sin)
    dimension = my_net.get_dimension(shape)

    neural_net.to(DEVICE)

    swarm = pso.Swarm(num=10,
                      dimension=dimension,
                      limit=[-0.5,
                             0.5],
                      omega=omega,
                      alpha_1=alpha1,
                      alpha_2=alpha2)

    # train
    prev_loss = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        for point, value in train_loader:
            running_loss += train(neural_net, point, value, swarm)
        else:
            # print(
            # f"free memory:
            # {torch.cuda.memory_cached(0)-torch.cuda.memory_allocated(0)}")
            training_loss = running_loss / len(train_loader)
            if print_epoch and training_loss < prev_loss:
                print(
                    f"epoch: {epoch}, training loss: {training_loss}")

            prev_loss = training_loss

    # record training loss

    # test
    my_net.update_weights(neural_net, swarm.g)
    testing_loss = 0
    for point, value in test_loader:
        testing_loss += test(neural_net, point, value)
    testing_loss /= len(test_loader)

    return (training_loss, training_loss, swarm.g.tolist())


if __name__ == '__main__':

    FILENAME = sys.argv[1]
    RAW_TRAIN = get_spiral_data(FILENAME + "/train.dat")
    RAW_TEST = get_spiral_data(FILENAME + "/test.dat")

    TRAIN, TEST = SpiralDataset(RAW_TRAIN), SpiralDataset(RAW_TEST)

    TRAIN_L = torch.utils.data.DataLoader(TRAIN, len(TRAIN))
    TEST_L = torch.utils.data.DataLoader(TEST, len(TEST))

    RESULT = train_and_test(
        TRAIN_L,
        TEST_L,
        epochs=5000,
        shape=[6, 8, 1],
        omega=0.6,
        alpha1=1.61,
        alpha2=1.61)

    OUT_RAW = {
        "training_loss": float(RESULT[0]),
        "testing_loss": float(RESULT[1]),
        "best_weight": RESULT[2]
    }

    OUT = json.dumps(OUT_RAW)
    print(OUT)
    sys.exit(0)

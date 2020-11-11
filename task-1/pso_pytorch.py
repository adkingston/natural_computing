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


def get_spiral_data():
    """ gets data set from file """
    out = []
    with open("two_spirals.dat") as data:
        for line in data:
            x1, x2, y = line.split()
            vec = torch.cuda.FloatTensor([float(x1), float(x2), float(
                x1)**2, float(x2)**2, np.sin(float(x1)), np.sin(float(x2))])
            y = torch.cuda.FloatTensor([float(y)])
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


@time_it
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
            point, value = point.to(DEVICE), value.to(DEVICE)
            neural_net.train()

            objective_func = my_net.objective(
                point, value, neural_net, my_net.squared_error)
            swarm.perform_iteration(objective_func)

            my_net.update_weights(neural_net, swarm.g)

            prediction = neural_net(point)
            loss = my_net.squared_error(prediction, value)
            running_loss += loss.item()
        else:
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

        point, value = point.to(DEVICE), value.to(DEVICE)
        prediction = neural_net(point)
        loss = my_net.squared_error(prediction, value)

        testing_loss += loss.item()

    testing_loss /= len(test_loader)

    return (training_loss, training_loss, swarm.g)


if __name__ == '__main__':

    DATA = get_spiral_data()

    # we will use cross validation to try and get a better understanding of the
    # true error of the model
    NUM_PARTS = 10
    PARTS = partition(DATA, NUM_PARTS)  # partition into 5 pieces

    AVERAGE_TRAIN_LOSS = 0.0
    AVERAGE_TEST_LOSS = 0.0
    for iteration in range(NUM_PARTS):
        print(f"========== ITERATION {iteration} =============")
        train, test = get_train_test_sets(PARTS, iteration)

        train, test = SpiralDataset(train), SpiralDataset(test)

        train_l = torch.utils.data.DataLoader(train, len(train))
        test_l = torch.utils.data.DataLoader(test, len(test))

        RESULT = train_and_test(
            train_l,
            test_l,
            epochs=5000,
            shape=[6, 8, 1],
            omega=0.6,
            alpha1=1.61,
            alpha2=1.61)
        print(
            f'iteration: {iteration}, training loss: {RESULT[0]}, testing loss: {RESULT[1]}')

        print("best weights:", RESULT[2])

        AVERAGE_TRAIN_LOSS += RESULT[0]
        AVERAGE_TEST_LOSS += RESULT[1]

    AVERAGE_TRAIN_LOSS /= NUM_PARTS
    AVERAGE_TEST_LOSS /= NUM_PARTS

    print(
        f'average training loss: {AVERAGE_TRAIN_LOSS}, average test loss: {AVERAGE_TEST_LOSS}')

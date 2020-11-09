"""
implements optimizer interface by pytorch
"""

from torch import nn
import torch
import pso
import numpy as np
from torch.utils.data import Dataset
import torch.optim
import matplotlib.pyplot as plt
import nn as my_net


class SpiralDataset(Dataset):
    def __init__(self, get_data):
        self.data = get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]


DEVICE = torch.device("cpu")


def get_training_data():
    out = []
    with open("two_spirals.dat") as data:
        for line in data:
            x1, x2, y = line.split()
            vec = torch.FloatTensor([float(x1), float(x2), np.sin(float(x1)),
                                     np.sin(float(x2))])
            vec = vec.to(DEVICE)
            y = torch.FloatTensor([float(y)]).to(DEVICE)
            out.append([vec, y])

    return out[:len(out) // 2]


def get_testing_data():
    out = []
    with open("two_spirals.dat") as data:
        for line in data:
            x1, x2, y = line.split()
            vec = torch.FloatTensor([float(x1), float(x2), np.sin(float(x1)),
                                     np.sin(float(x2))])
            vec = vec.to(DEVICE)
            y = torch.FloatTensor([float(y)]).to(DEVICE)
            out.append([vec, y])

    return out[len(out) // 2:]


TRAINING_DATA = SpiralDataset(get_training_data)
TESTING_DATA = SpiralDataset(get_testing_data)
# fig = plt.figure()
# for i in range(len(SPIRAL_DATA)):
# if SPIRAL_DATA[i][1] == 1:
# color = 'ro'
# else:
# color = 'bo'
# plt.plot(SPIRAL_DATA[i][0][0], SPIRAL_DATA[i][0][1], color)

# plt.show()

TRAINLOADER = torch.utils.data.DataLoader(TRAINING_DATA, batch_size=10)
TESTLOADER = torch.utils.data.DataLoader(TESTING_DATA, batch_size=4)

SHAPE = [4, 1]
NET = my_net.new_net(shape=SHAPE, activator=nn.Tanh)
DIMENSION = my_net.get_dimension(SHAPE)
NET.to(DEVICE)
SWARM = pso.Swarm(num=25, dimension=DIMENSION, limit=[-100000, 100000],
                  omega=0.5, alpha_1=2.02, alpha_2=2.02)

print("=============== TRAINING ===============")
EPOCHS = 100
for e in range(EPOCHS):
    running_loss = 0.0
    index = 0
    for point, value in TRAINLOADER:
        point, value = point.to(DEVICE), value.to(DEVICE)
        NET.train()

        objective_func = my_net.objective(
            point, value, NET, my_net.squared_error)
        SWARM.perform_iteration(objective_func)

        my_net.update_weights(NET, SWARM.global_best)

        prediction = NET(point)
        loss = my_net.squared_error(prediction, value)
        running_loss += loss.item()
        print(
            f"batch no: {index}, current loss: {loss.item()}, batch running loss: {running_loss}")
        index += 1
    else:
        print(f"training loss: {running_loss/len(TRAINLOADER)}")


print("=============== TESTING ===============")
# ensure the final global best is set
my_net.update_weights(NET, SWARM.global_best)

for e in range(5):
    testing_loss = 0.0
    for point, value in TESTLOADER:
        point, value = point.to(DEVICE), value.to(DEVICE)
        prediction = NET(point)
        loss = my_net.squared_error(prediction, value)

        testing_loss += loss.item()
    else:
        print(f"testing loss: {testing_loss/len(TESTLOADER)}")

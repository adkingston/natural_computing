"""
implements optimizer interface by pytorch
"""

from torch import nn
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.optim
import matplotlib.pyplot as plt


class SpiralDataset(Dataset):
    def __init__(self):
        self.data = get_training_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]


DEVICE = torch.device("cuda:0")


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

    return out


SPIRAL_DATA = SpiralDataset()
print(len(SPIRAL_DATA))
# fig = plt.figure()
# for i in range(len(SPIRAL_DATA)):
# if SPIRAL_DATA[i][1] == 1:
# color = 'ro'
# else:
# color = 'bo'
# plt.plot(SPIRAL_DATA[i][0][0], SPIRAL_DATA[i][0][1], color)

# plt.show()

TRAINLOADER = torch.utils.data.DataLoader(SPIRAL_DATA, batch_size=16)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 6)
        self.fc3 = nn.Linear(6, 2)
        self.fc4 = nn.Linear(2, 1)

    def forward(self, x):
        relu = nn.Tanh()
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        return x


NET = NeuralNetwork()
NET.to(DEVICE)
CRITERION = nn.MSELoss()
OPTIMIZER = torch.optim.SGD(NET.parameters(), lr=0.1)

EPOCHS = 5000
for e in range(EPOCHS):
    running_loss = []
    i = 0
    for point, value in TRAINLOADER:
        point, value = point.to(DEVICE), value.to(DEVICE)
        NET.train()

        prediction = NET(point)
        # prediction = torch.reshape(prediction, (-1, ))
        loss = CRITERION(prediction, value)
        OPTIMIZER.zero_grad()
        # value = value.float()
        loss.backward()
        OPTIMIZER.step()

        running_loss.append(loss.item())
    else:
        # print(f"Training loss: {running_loss[-1]/len(TRAINLOADER)}")
        l = sum(running_loss)
        print(f"training loss: {l/len(TRAINLOADER)}")

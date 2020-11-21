"""
helper functions for setting up data for neural networks
"""

import torch
import numpy as np
from torch.utils.data import Dataset


def get_dimension(shape=(2, 1)):
    s = 0
    for i in range(len(shape) - 1):
        s += shape[i + 1] * (shape[i] + 1)
    return s - shape[-1]


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


def get_spiral_data(filename, transform, tr=False):
    """ gets data set from file """
    out = []
    with open(filename) as data:
        for line in data:
            x1, x2, y = line.split()
            vec = transform(x1, x2, tr)
            y = np.float64(y)

            if tr:
                y = torch.FloatTensor([y])
            else:
                y = np.array([y])

            out.append([vec, y])

    return out


def T6(x1, x2, tr=False):
    retval = [float(x1),
              float(x2),
              # float(x1)**2,
              # float(x2)**2,
              np.sin(float(x1)),
              np.sin(float(x2))]

    if tr:
        return torch.FloatTensor(retval)
    return np.array(retval)


def T2(x1, x2, tr=False):
    retval = [np.float64(x1), np.float64(x2)]
    if tr:
        return torch.FloatTensor(retval)
    return np.array(retval)

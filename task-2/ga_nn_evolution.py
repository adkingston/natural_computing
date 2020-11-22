#!/usr/bin/env python
# coding: utf-8

import itertools
from torch import nn
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.optim
import json
from gen_algorithm import geneticalgorithm as ga


DEVICE = torch.device("cpu") 

def get_data_torch():
    out = []
    with open("c:/UOE/NaturalComputing/natural_computing/task-1/two_spirals.dat") as data:
        for line in data:
            x1, x2, label = line.split()
            x1=float(x1)
            x2=float(x2)
            label=float(label)
            
            vec = torch.FloatTensor([x1, x2, np.sin(x1),
                                     np.sin(x2)])
            vec = vec.to(DEVICE)
            y = torch.FloatTensor([label]).to(DEVICE)
            out.append([vec, y])
    datalen=int(len(out)/2)
    return out[:datalen],out[datalen:]

training_data, testing_data = get_data_torch()


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
        #x = nn.functional.sigmoid(self.lin_layers[-1](x))

        return x

    def add_layers(self):
        # adjust nn_list
        length = len(self.nn_list)

        # adjust lin_layers
        for i in range(length-1):
            self.lin_layers.append(nn.Linear(self.nn_list[i], self.nn_list[i+1]))
        
#test       
#net1 = NeuralNetwork([4,5,6,2,1])
#print('First Network:::')
#for p in net1.parameters():
#    print(p)


TRAINLOADER = torch.utils.data.DataLoader(training_data, batch_size=160)
TESTLOADER = torch.utils.data.DataLoader(testing_data, batch_size=160)


def test_nn(nn_shape):
    net = NeuralNetwork(nn_shape)
    net.to(DEVICE)
    OPTIMIZER = torch.optim.SGD(net.parameters(), lr=0.1)
    EPOCHS = 10000
    CRITERION = nn.MSELoss()
    
    for e in range(EPOCHS):
        train_loss = 0
        test_loss = 0

        for point, value in TRAINLOADER:
            point, value = point.to(DEVICE), value.to(DEVICE)
            net.train()
            prediction = net(point)
            
            loss = CRITERION(prediction, value)
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            train_loss += loss.item()

        for point, value in TESTLOADER:
            point, value = point.to(DEVICE), value.to(DEVICE)
        
            with torch.no_grad():
                prediction = net.forward(point)
                loss2 = CRITERION(prediction, value)
                test_loss += loss2.item()
        
        if (test_loss-train_loss)>0.1:
            #print('here: ',train_loss,test_loss)
            train_loss = 1000
            break
        if test_loss<0.01:
            break
        #if e % 100 == 99:
        #    print(train_loss,test_loss)
    
    return (train_loss+test_loss)/2

    #prediction[prediction>0.5]=1
    #prediction[prediction<=0.5]=0
    #print(prediction)

#test_nn([4,5,6,2,1])



def ga_error(X):
    X = [int(i) for i in X if int(i) != 0] #Drop all zeroes
    X.insert(0,4) #add 4, the number of input features, at the beginning
    X.append(1) #we just want 1 output
    
    #Build in some redundancy to account for poor initial values
    err = [test_nn(X) for i in range(2)]
        
    error = (min(err)*1000) + np.sum(X[1:-1]) + 2*len(X)

    return error

#print(ga_error([5,6,2]))
#print(ga_error([6,2]))
#print(ga_error([2,2]))
#print(ga_error([6,6,6,2]))


algorithm_param = {'max_num_iteration': 100,
                   'population_size':30,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.2,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None}

varbound=np.array([[0,8]]*4)

model=ga(function=ga_error,
         dimension=4,
         variable_type='int',
         variable_boundaries=varbound,
         algorithm_parameters=algorithm_param,
         function_timeout=200)

model.run()

convergence=model.report
#solution=model.ouput_dict
#print(convergence)

f = open("GA_output.txt", "a")
f.write(json.dumps(algorithm_param))
for item in convergence:
    f.write("%s\n" % item)
f.write("\n\n")
f.close()


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
    with open("two_spirals.dat") as data:
        for line in data:
            x1, x2, label = line.split()
            x1=float(x1)
            x2=float(x2)
            label=float(label)
            
            vec = torch.FloatTensor([x1, x2, x1*x1, x2*x2, np.sin(x1),
                                     np.sin(x2)])
            vec = vec.to(DEVICE)
            y = torch.FloatTensor([label]).to(DEVICE)
            out.append([vec, y])
    datalen=int(len(out)/2)
    return out[:datalen],out[datalen:]

training_data, testing_data = get_data_torch()


activation_funcs = [torch.tanh, torch.sin, torch.sigmoid]

class NeuralNetwork(nn.Module):
    '''
    Args:
        nn_list = [input dim., first hidden layer size,...,last hidden layer size, output dim.]

    '''
    def __init__(self, nn_list, activation_func):
        super(NeuralNetwork, self).__init__()
        self.nn_list = nn_list
        self.activation_func = activation_func
        self.lin_layers = nn.ModuleList()
        self.add_layers()

    def forward(self, x):
        # forward pass through the network
        for layer in self.lin_layers:
            x = self.activation_func(layer(x))
        
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
net1 = NeuralNetwork([4,5,1], activation_funcs[3])
print('First Network:::')
for p in net1.parameters():
    print(p)
    
TRAINLOADER = torch.utils.data.DataLoader(training_data, batch_size=160)
TESTLOADER = torch.utils.data.DataLoader(testing_data, batch_size=160)

learning_rates = [0.001, 0.01 , 0.03, 0.1, 0.3, 0.5]
inputs = [2, 4, 6]

def test_nn(nn_shape, activation_func, learning_rate):
    net = NeuralNetwork(nn_shape,activation_func)
    net2 = NeuralNetwork(nn_shape,activation_func)

    net.to(DEVICE)
    net2.to(DEVICE)

    OPTIMIZER = torch.optim.SGD(net.parameters(), lr=learning_rate)
    OPTIMIZER2 = torch.optim.SGD(net2.parameters(), lr=learning_rate)

    EPOCHS = 5000
    CRITERION = nn.MSELoss()

    for e in range(EPOCHS):
        train_loss = 0
        train_loss2 = 0

        test_loss = 0
        test_loss2 = 0

        for point, value in TRAINLOADER:
            point, value = point.to(DEVICE), value.to(DEVICE)
            net.train()
            net2.train()

            prediction = net(point)
            prediction2 = net2(point)
            
            loss = CRITERION(prediction, value)
            loss2 = CRITERION(prediction2, value)

            OPTIMIZER.zero_grad()
            OPTIMIZER2.zero_grad()

            loss.backward()
            loss2.backward()

            OPTIMIZER.step()
            OPTIMIZER2.step()

            train_loss += loss.item()
            train_loss2 += loss2.item()
        
        if e % 100 == 99:
            for point, value in TESTLOADER:
              point, value = point.to(DEVICE), value.to(DEVICE)
        
              with torch.no_grad():
                prediction = net.forward(point)
                prediction = net.forward(point)

                loss_test = CRITERION(prediction, value)
                loss_test2 = CRITERION(prediction2, value)

                test_loss += loss_test.item()
                test_loss2 += loss_test2.item()

            #if (test_loss-train_loss)>0.1:
            #  #print('here: ',train_loss,test_loss)
            #  train_loss = 1000
            #  break
            if test_loss<0.01 or test_loss2<0.01:
              break

            print(train_loss,test_loss,train_loss2,test_loss2)
    
    return min([(train_loss+test_loss)/2,
                (train_loss2+test_loss2)/2])

    #prediction[prediction>0.5]=1
    #prediction[prediction<=0.5]=0
    #print(prediction)

test_nn([6,8,2,1])

def ga_error(X):
    X = [int(i) for i in X[:4] if int(i) != 0] #Drop all zeroes
    X.insert(0,6) #add 4, the number of input features, at the beginning
    X.append(1) #we just want 1 output
    
    #Build in some redundancy to account for poor initial values
    err = [test_nn(X) for i in range(3)]
        
    error = (min(err)*1000) + np.sum(X[1:-1]) + 2*len(X)

    return error

#print(ga_error([5,6,2]))
#print(ga_error([6,2]))
#print(ga_error([2,2]))
#print(ga_error([6,6,6,2]))

algorithm_param = {'max_num_iteration': 2,
                   'population_size':3,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None}

varbound=np.array([[0,8],[0,8],[0,8],[0,8],[0,2],[0,5]])

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


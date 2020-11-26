#!/usr/bin/env python
# coding: utf-8

import itertools
from torch import nn
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.optim
import json
from gen_algorithm import geneticalgorithm

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

#Run the neural network with a given shape twice and return the best 
#testing+training loss
def test_nn(nn_shape):
    net = NeuralNetwork(nn_shape)
    net2 = NeuralNetwork(nn_shape)

    net.to(DEVICE)
    net2.to(DEVICE)

    OPTIMIZER = torch.optim.SGD(net.parameters(), lr=0.3)
    OPTIMIZER2 = torch.optim.SGD(net2.parameters(), lr=0.3)

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

            if (test_loss-train_loss)>0.1:
            #  #print('here: ',train_loss,test_loss)
              train_loss = 1000
              break
            if test_loss<0.01 or test_loss2<0.01:
              #print(train_loss,test_loss,train_loss2,test_loss2)
              #print('hererererer')
              break

            #print(train_loss,test_loss,train_loss2,test_loss2)
    
    return min([(train_loss+test_loss)/2,
                (train_loss2+test_loss2)/2])

    #prediction[prediction>0.5]=1
    #prediction[prediction<=0.5]=0
    #print(prediction)

#for i in range(0,100):
#  test_nn([6,8,1])

#Calculate the fitness for the genetic algorithm and also append
#the input and ouput shape
def ga_error(X):
    X = [int(i) for i in X if int(i) != 0] #Drop all zeroes
    X.insert(0,6) #add 4, the number of input features, at the beginning
    X.append(1) #we just want 1 output
    
    #Build in some redundancy to account for poor initial values
    #err = [test_nn(X) for i in range(2)]
        
    error = (test_nn(X)*1000) + np.sum(X[1:-1]) + 2*len(X)

    return error

#print(ga_error([5,6,2]))
#print(ga_error([6,2]))
#print(ga_error([2,2]))
#print(ga_error([6,6,6,2]))

initial_pop_30 = np.array([[4.,7.,5.,0., 32.47409289],
 [4.,2.,7.,0., 75.47345939],
 [1.,5.,3.,4.,172.43256569],
 [5.,1.,8.,3., 99.20322978],
 [8.,4.,1.,2., 39.54992187],
 [7.,3.,0.,5., 33.97280872],
 [6.,4.,7.,7., 58.87075156],
 [2.,8.,3.,4.,174.60700953],
 [8.,3.,7.,8., 46.27320758],
 [5.,0.,8.,8., 88.60540068],
 [5.,8.,2.,0., 30.61208394],
 [3.,8.,1.,0.,119.77797386],
 [7.,5.,6.,7., 46.46126878],
 [8.,6.,5.,5., 44.75294209],
 [2.,6.,5.,8.,128.33540905],
 [0.,3.,2.,5.,144.33516234],
 [6.,0.,8.,5., 50.39336895],
 [6.,6.,3.,0., 30.24390582],
 [7.,1.,0.,2.,103.70982856],
 [3.,2.,8.,0.,136.59323561],
 [0.,3.,1.,3.,115.34284708],
 [4.,8.,4.,1., 59.92755936],
 [2.,5.,7.,7.,201.09097677],
 [7.,5.,5.,1., 53.69114943],
 [0.,5.,8.,3., 51.37153754],
 [1.,2.,6.,8.,210.35812879],
 [5.,1.,7.,5., 38.60409904],
 [4.,6.,4.,4., 39.94563103],
 [7.,2.,1.,5., 36.973648  ],
 [5.,3.,5.,1., 93.34599173]])

algorithm_param = {'max_num_iteration': 100,
                   'population_size':30,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None}

varbound=np.array([[0,8]]*4)

model=geneticalgorithm(function=ga_error,
         dimension=4,
         variable_type='int',
         variable_boundaries=varbound,
         algorithm_parameters=algorithm_param,
         initial_pop=initial_pop_30,
         function_timeout=200)

model.run()

convergence=model.report
print(convergence)

with open('GA_output.txt', 'w') as f:
  f.write(json.dumps(algorithm_param))
  f.write(np.array2string(model.best_variable))
  for item in convergence:
    f.write("%s\n" % item)
  f.write("\n\n")

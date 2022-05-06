#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import *

import itertools

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

from .values import ValueFunction
from .features import FeatureVector


class Net(torch.nn.Module):
    def __init__(self,
        n_feature, n_hidden=32, n_output=1, # number of nodes
        hidden_layers=2, # number of layers
        hidden_activation=F.relu, predict_activation=lambda x: x, # activation functions
    ):
        super(Net, self).__init__()
        n_hidden = np.broadcast_to(n_hidden, (hidden_layers))
        self.hidden = []
        for i,(m,n) in enumerate(zip(itertools.chain([n_feature],n_hidden),n_hidden)):
            h = torch.nn.Linear(m, n)
            self.hidden.append(h)
            setattr(self, f'hidden{i+1}', h) # register layer
        self.predict = torch.nn.Linear((n_hidden[-1] if n_hidden.size else n_feature), n_output)
        self.hidden_activation = np.broadcast_to(hidden_activation, (hidden_layers))
        self.predict_activation = predict_activation

    def forward(self, x):
        for hidden, activation in zip(self.hidden, self.hidden_activation):
            x = activation(hidden(x))
        x = self.predict_activation(self.predict(x))
        return x



class NNValueFunctionFromFeatureVector(ValueFunction):
    def __init__(self,
        X:FeatureVector,
        alpha = 0.001,
    ):
        """
        state_dims: the number of dimensions of state space
        """
        # self.net = Net(n_feature=state_dims, n_hidden=32, n_output=1, n_layers=2)     # define the network
        self.X = X
        self.net = Net(n_feature=X.size)     # define the network
        # print(self.net); exit()
        self.optimizer = torch.optim.Adam(self.net.parameters(),
            lr = alpha,
            betas = [0.9, 0.999],
        )
        self.loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    def __call__(self, state):
        x = self.X[state]
        x = torch.from_numpy(x).float()

        self.net.eval()

        v = self.net(x)
        return v.detach().numpy()[0]

    def update(self, state, G, alpha=None):
        # if alpha is not None: raise NotImplementedError()
        x = self.X[state]
        x = torch.from_numpy(x).float()
        G = torch.from_numpy(np.array([G])).float()
        # G = torch.tensor(G).float()

        self.net.train()

        prediction = self.net(x)     # input x and predict based on x
        loss = self.loss_func(prediction, G)     # must be (1. nn output, 2. target)
        self.optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        self.optimizer.step()        # apply gradients
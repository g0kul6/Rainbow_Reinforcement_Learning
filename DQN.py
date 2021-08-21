#for Deep Q-learning network
from parameter import env,action_size,state_size,replay_size

#importing necessary packages
import copy
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#creating the DQN agent
class DQN_agent:
    def __init__(self,layer_sizes,alpha,sync_freq,replay_size):
        #intializing q_net and target_net
        self.q_net=self.net_nn(layer_sizes).cuda()
        self.target_net=copy.deepcopy(self.q_net).cuda()
        #loss function as mean squared function
        self.loss_function=nn.MSELoss
        #optimizer
        self.optim=nn.Adam(self.q_net.parameters(),alpha)
        self.gamma=torch.tensor(0.95).float().cuda()
        



    #building the nn model
    def net_nn(self,layer_sizes):
        #layers list
        layers=[]
        for i in range(len(layer_sizes)-1):
            linear=nn.Linear(layer_sizes[i],layer_sizes[i+1])
            activation=nn.Tanh() if i<len(layer_sizes)-2 else nn.Identity()
            layers+=(linear,activation)
        return nn.Sequential(layers)







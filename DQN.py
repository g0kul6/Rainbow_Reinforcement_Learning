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
        #sync for updating target network
        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        #discount factor
        self.gamma=torch.tensor(0.95).float().cuda()
        self.experience_replay = deque(maxlen = replay_size) 

    #building the nn model
    def net_nn(self,layer_sizes):
        #layers list
        layers=[]
        for i in range(len(layer_sizes)-1):
            linear=nn.Linear(layer_sizes[i],layer_sizes[i+1])
            activation=nn.Tanh() if i<len(layer_sizes)-2 else nn.Identity()
            layers+=(linear,activation)
        return nn.Sequential(layers)

    #getting the action using epsilon 
    def get_action(self,state,action_size,epsilon):
        with torch.no_grad():
            Q_predicted=self.q_net(torch.from_numpy(state).float().cuda())
        Q,A=torch.max(Q_predicted,axis=0)
        A=A if torch.rand(1,).item() > epsilon else torch.randint(0,action_size,(1,))
        return A
    
    #get next q    
    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q,_ = torch.max(qp, axis=1)    
        return q

    #collecting experience 
    def collect_experience(self, experience):
        self.experience_replay.append(experience)

    #sample from experience
    




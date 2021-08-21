#for Deep Q-learning network
from parameter import env,action_size,state_size

#importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#creating the neural net
class Network(nn.Module):
    def __init__(self,dim_in,dim_out):
        #intializing the model
        super(Network, self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(dim_in,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.Linear(128,dim_out)
        )
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        #forward prop
        return self.layers(x)






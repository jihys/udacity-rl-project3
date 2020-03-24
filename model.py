
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def custom_hidden_init(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)        

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=2, actor_bn1=True, actor_bn2=True, custom_init=True,fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
 
        
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_units)       
        self.fc2 = nn.Linear(fc1_units, fc2_units)       
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.actor_bn1=actor_bn1
        self.actor_bn2=actor_bn2
        
        
        
        if actor_bn1:
            self.bn1=nn.BatchNorm1d(state_size)
        if actor_bn2:
            self.bn2=nn.BatchNorm1d(fc1_units)
            self.bn3=nn.BatchNorm1d(fc2_units) 
            
        if custom_init:
            custom_hidden_init(self.fc1)
            custom_hidden_init(self.fc2)
            custom_hidden_init(self.fc3)
        else:
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        """
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        
        print("x state:{}".format(x.shape)) 
        """
        x=state
        #print("x state:{}".format(x.shape)) 
        if self.actor_bn1:
            x = self.bn1(x)
        x = F.relu(self.fc1(x))
        if self.actor_bn2:
            x = self.bn2(x)
        x = F.relu(self.fc2(x))
        if self.actor_bn2:
            x = self.bn3(x)
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed=2,critic_bn1=True, critic_bn2=True,custom_init=True, fcs1_units=256, fc2_units=128,):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)      
        self.fcs1 = nn.Linear(state_size, fcs1_units)        
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.critic_bn1=critic_bn1
        self.critic_bn2=critic_bn2
        self.custom_init=custom_init
        
        if critic_bn1:
            self.bn1=nn.BatchNorm1d(state_size)
        if critic_bn2:
            self.bn2=nn.BatchNorm1d(fcs1_units)
            
        if custom_init:
            custom_hidden_init(self.fcs1)
            custom_hidden_init(self.fc2)
            custom_hidden_init(self.fc3)
        else:
            self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc3.weight.data.uniform_(-3e-4, 3e-4)   # Value from DDPG paper

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
      
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x=state
        if self.critic_bn1:
            x=self.bn1(x)
        x = F.relu(self.fcs1(x))
        if self.critic_bn2:
            x=self.bn2(x)
        xs = torch.cat((x, action), dim=1)
        xs= F.relu(self.fc2(xs))
        return self.fc3(xs)

import torch
from torch.optim import Adam

from model import Actor, Critic
from noise import OUNoise
import random

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
#WEIGHT_DECAY = 1e-2        # L2 weight decay (Value from DDPG paper)
WEIGHT_DECAY = 0

NOISE_START=1.0
NOISE_END=0.1
NOISE_REDUCTION=0.999
EPISODES_BEFORE_TRAINING = 300


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class DDPGAgent:

    def __init__(self,num_agents, state_size, action_size, random_seed,actor_bn1, actor_bn2,critic_bn1, critic_bn2,custom_init,fc1_units, fc2_units):
       
 
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents=num_agents
        self.actor_bn1=actor_bn1
        self.actor_bn2=actor_bn2
        self.critic_bn1=critic_bn1
        self.critic_bn2=critic_bn2
        self.custom_init=custom_init
        self.fc1_units=fc1_units
        self.fc2_units=fc2_units   
    
      
        ##Create Actor & Critic networks
        
        self.actor = Actor(state_size, action_size, random_seed,actor_bn1,actor_bn2,custom_init,fc1_units,fc2_units).to(device)
        self.target_actor = Actor(state_size, action_size, random_seed,actor_bn1,actor_bn2,custom_init,fc1_units,fc2_units).to(device)

        self.critic = Critic(state_size * num_agents, action_size * num_agents,random_seed,critic_bn1, critic_bn2,custom_init,fc1_units,fc2_units).to(device)
        self.target_critic = Critic(state_size * num_agents, action_size * num_agents,random_seed,critic_bn1, critic_bn2,custom_init,fc1_units,fc2_units).to(device)

        self.noise = OUNoise(device, action_size)
        self.noise_scale = NOISE_START

        self.actor_optimizer = Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.hard_update()
        
        

    def act(self, states, target=False, noise=0.0,train=False):
      
        actor_network = self.target_actor if target else self.actor
        
        if not train:
            actor_network.eval()
        action = actor_network(states)
        
        if not train:
            actor_network.train()
         
        if noise != 0:
            #self.noise_scale=max(NOISE_END,NOISE_REDUCTION**i_episode) 
            noisy_action = action + noise * self.noise.noise()
            return noisy_action.clamp(-1, 1)

        return action
    
  
                                   
     
    
    
    
    
    
    
    def reset(self):
        self.noise.reset()

    def hard_update(self):
        """Performs a hard update on the target networks (copying the values from the local networks). """
        DDPGAgent._hard_update(self.target_actor, self.actor)
        DDPGAgent._hard_update(self.target_critic, self.critic)

    def soft_update(self, tau):
        """  Performs a soft update on the target networks:
            target_params := target_params * (1 - tau) + local_params * tau

        :param tau: (float) Update scaling parameter
        """
        DDPGAgent._soft_update(self.target_actor, self.actor, tau)
        DDPGAgent._soft_update(self.target_critic, self.critic, tau)

    def _hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)            
            
    
    def _soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


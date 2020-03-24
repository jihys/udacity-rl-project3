import torch
import torch.nn.functional as F

import numpy as np

from ddpg import DDPGAgent
from memory import ReplayBuffer
import random

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
#WEIGHT_DECAY = 1e-2        # L2 weight decay (Value from DDPG paper)
WEIGHT_DECAY = 0
gradient_clip_actor=1.0
gradient_clip_critic=1.0





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class MADDPG:

    def __init__(self,num_agents, state_size, 
            action_size, random_seed=2,
            actor_bn1=True,actor_bn2=True,
            critic_bn1=True, critic_bn2=True,
            custom_init=False,fc1_units=256, fc2_units=128,update_every=1):

       

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        
        self.tau = TAU
        self.gradient_clip_actor = gradient_clip_actor
        self.gradient_clip_critic = gradient_clip_critic                
        self.seed = random.seed(random_seed)       
        self.actor_bn1=actor_bn1
        self.actor_bn2=actor_bn2
        self.critic_bn1=critic_bn1
        self.critic_bn2=critic_bn2
        self.custom_init=custom_init
        self.fc1_units=fc1_units
        self.fc2_units=fc2_units
        self.update_every=update_every
        self.t_step = 0
        self.episode = 0
        self.agents = []
        for i in range(num_agents):
            self.agents.append(DDPGAgent(
                  num_agents, state_size, action_size,random_seed,actor_bn1=self.actor_bn1, 
                actor_bn2=self.actor_bn2,critic_bn1=self.critic_bn1,
                critic_bn2=self.critic_bn2, custom_init=self.custom_init, 
                fc1_units=self.fc1_units, fc2_units=self.fc2_units))

        # Replay memory
        self.memory = ReplayBuffer(
            buffer_size=BUFFER_SIZE,
            device=device
        )
    def reset_agents(self):
        for agent in self.agents:
            agent.reset()
    
    def step(self, state, action, reward, next_state, done):
        """ Store a single agent step, learning every N steps

         :param state: (array-like) Initial states on the visit
         :param action: (array-like) Actions on the visit
         :param reward: (array-like) Rewards received on the visit
         :param next_state:  (array-like) States reached after the visit
         :param done:  (array-like) Flag whether the next states are terminal states
         """

        self.memory.add(state, action, reward, next_state, done)

        # Learn every self.update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random batch and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences)

        # Keep track of episode number
        if np.any(done):
            self.episode += 1

    def act(self, states, target=False, noise=1.0):
        """ Returns the selected actions for the given states according to the current policy

        :param states: (array-like) Current states
        :param target:  (boolean, default False) Whether to use local networks or target networks
        :param noise:  (float, default 1)  Scaling parameter for noise process
        :return: action (array-like)  List of selected actions
        """

        if type(states) == np.ndarray:
            states = torch.from_numpy(states).float().to(device)

        actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                agent = self.agents[i]
                action = agent.act(states[i, :].view(1, -1),target=target, noise=noise)
                actions.append(action.squeeze())
        actions = torch.stack(actions)

        return actions.cpu().data.numpy()

    def learn(self, experiences):
        """ Performs training for each agent based on the selected set of experiencecs

        :param experiences:   Batch of experience tuples (s, a, r, s', d) collected from the replay buffer
        """

        state, action, rewards, next_state, done = experiences

        state = state.view(-1, self.num_agents, self.state_size)
        action = action.view(-1, self.num_agents, self.action_size)
        rewards = rewards.view(-1, self.num_agents)
        next_state = next_state.view(-1, self.num_agents, self.state_size)
        done = done.view(-1, self.num_agents)

        # Select agent being updated based on ensemble at time of samples
        for agent_number in range(self.num_agents):
            agent = self.agents[agent_number]

            # Compute the critic loss
            target_actions = []
            for i in range(self.num_agents):
                i_agent = self.agents[i]
                #print(next_state[:,i,:].shape)
                i_action = i_agent.act(next_state[:, i, :].contiguous(), target=True, noise=0.0, train=True)
                target_actions.append(i_action.squeeze())
            target_actions = torch.stack(target_actions)
            target_actions = target_actions.permute(1, 0, 2).contiguous()

            with torch.no_grad():
                flat_next_state = next_state.view(-1, self.num_agents * self.state_size)
                flat_target_actions = target_actions.view(-1, self.num_agents * self.action_size)
                Q_targets_next = agent.target_critic(flat_next_state, flat_target_actions).squeeze()

            Q_targets = rewards[:, agent_number] + GAMMA * Q_targets_next * (1 - done[:, agent_number])

            flat_state = state.view(-1, self.num_agents * self.state_size)
            flat_action = action.view(-1, self.num_agents * self.action_size)
            Q_expected = agent.critic(flat_state, flat_action).squeeze()

            critic_loss = F.mse_loss(Q_targets, Q_expected)

            # Minimize the critic loss
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), self.gradient_clip_critic)
            agent.critic_optimizer.step()

            # Compute the actor loss
            Q_input = []
            for i in range(self.num_agents):
                i_agent = self.agents[i]
                Q_input.append(i_agent.actor(state[:, i, :].contiguous()))
            Q_input = torch.stack(Q_input)
            Q_input = Q_input.permute(1, 0, 2).contiguous()
            flat_Q_input = Q_input.view(-1, self.num_agents * self.action_size)

            actor_loss = -agent.critic(flat_state, flat_Q_input).mean()

            # Minimize the actor loss
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), self.gradient_clip_actor)
            agent.actor_optimizer.step()

            # soft update target
            agent.soft_update(self.tau)

            
    def save(self, filename,model_num):
        """Saves the model networks to a file.

        :param filename:  Filename where to save the networks
        """
        checkpoint = {}
        for index, agent in enumerate(self.agents):
            checkpoint['actor_' +str(model_num)+ str(index)] = agent.actor.state_dict()
            checkpoint['target_actor_' +str(model_num)+ str(index)] = agent.target_actor.state_dict()
            checkpoint['critic_' +str(model_num)+str(index)] = agent.critic.state_dict()
            checkpoint['target_critic_' +str(model_num) + str(index)] = agent.target_critic.state_dict()
        torch.save(checkpoint, filename)


    def load(self, filename):
        """Loads the model networks from a file.

        :param filename: Filename from where to load the networks
        """
        checkpoint = torch.load(filename)

        for i in range(self.num_agents):
            agent = self.agents[i]
            agent.actor.load_state_dict(checkpoint['actor_' +str(model_num)+ str(index)])
            agent.target_actor.load_state_dict(checkpoint['target_actor_' +str(model_num)+ str(index)])
            agent.critic.load_state_dict(checkpoint['critic_' +str(model_num)+ str(index)])
            agent.target_critic.load_state_dict(checkpoint['target_critic_' +str(model_num) + str(index)])

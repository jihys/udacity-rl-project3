[//]: # "Image References"

[image1]: https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png "Trained Agent"



# Project 3: Build agents that play Tennis 

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.  Unity Machine Learning Agents (ML-Agents) plugin will be used to serve as environments for training intelligent agents. You can read more about ML-Agents by perusing the [GitHub repository](https://github.com/Unity-Technologies/ml-agents)

![Trained Agent][image1]



### Environment 

In this environment, two agents control rackets to bounce a ball over a  net. If an agent hits the ball over the net, it receives a reward of  +0.1.  If an agent lets a ball hit the ground or hits the ball out of  bounds, it receives a reward of -0.01.  Thus, the goal of each agent is  to keep the ball in play.

The observation space consists of 8 variables corresponding to the  position and velocity of the ball and racket. Each agent receives its  own, local observation.  Two continuous actions are available,  corresponding to movement toward (or away from) the net, and jumping. 



### Goal: Solving the Environment

The task is episodic, and in order to solve the environment, your  agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received  (without discounting), to get a score for each agent. This yields 2  (potentially different) scores. We then take the maximum of these 2  scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5



![Scores plot from the solution code.](https://video.udacity-data.com/topher/2018/August/5b75ef77_screen-shot-2018-08-16-at-4.37.07-pm/screen-shot-2018-08-16-at-4.37.07-pm.png)



### Getting Started

#### Install packages and dependencies

For the Training, I used a p2.xlarge type AWS EC2 instance (Ubuntu based Deep Learning AMI, AMI ID i-0b78e1a0328a9a1a6) the Seoul Region, where I closely located.  Most of the utilities are already installed  in Deep learning AMI so minor correction was made on requirment.txt file. 



1. Create and activate a new conda environment 

   ###### Linux or Mac:		

   ```
   conda create --name drlnd python=3.6
   source activate drlnd
   ```

   ###### Windows:

   ```
   conda create --name drlnd python=3.6
   activate drlnd
   ```

   

3. Clone following repository and install additional dependencies.

   ```
   git clone https://github.com/jihys/udacity-rl-project3.git
   
   #Alterntively you can download original codes from udcity github. 
   #git clone https://github.com/udacity/deep-reinforcement-learning.git`
   cd python`
   vi requirement.txt` 
   #"Correct requirement.txt as needed"`
   pip install .
   ```
   
   
   

#### Unity Environment Setup 

For this project, you will **not** need to install Unity - this is because we have already built the environment for you, and  you can download it from one of the links below.  You need only select  the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the `p3_collab-compet/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(*For Windows users*) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit  version or 64-bit version of the Windows operating system.

(*For AWS*) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (*To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.*)

 

### Instructions

After you have followed the instructions above, open `Tennis.ipynb` and follow the instructions to train the agents.

 


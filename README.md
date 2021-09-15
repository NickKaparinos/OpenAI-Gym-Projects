# OpenAI-Gym-Projects

# Gym
Gym is an open source Python library for developing and comparing reinforcement learning algorithms by providing a
standard API to communicate between learning algorithms and environments, as well as a standard set of environments 
compliant with that API. Since its release, Gym's API has become the field standard for doing this.

# Classic Control
Control theory problems from the classic RL literature.

## CartPole-v1
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. 
The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
A reward of +1 is provided for every timestep that the pole remains upright and the maximum number of steps per episode is 500. Hence,
a perfect agent would be able to achieve a reward of +500 every episode.

### Solution using Proximal Policy Optimization (**PPO**)

<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/CartPole/results/learning_curve.png" alt="drawing" width="500"/></p>

#### Agent after 1250 episodes
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/CartPole/results/ppo.gif" width="400"/></p>

## MountainCar-v0
A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum. A reward of -1 is provided for every timestep until the goal is reached or 200 timesteps have passed.


### Solution using Double Dueling DQN with Prioritized Experience Replay


<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/MountainCar/results/learning_curve100.png" alt="drawing" width="500"/></p>

#### Agent after 4000 episodes
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/MountainCar/results/openaigym.video.26.24298.video000000.gif" width="400"/></p>

## MountainCarContinuous-v0
A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum. Here, the reward is greater if you spend less energy to reach the goal


### Solution using Soft Actor-Critic (SAC) with Prioritized Experience Replay


<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/MountainCarContinuous/results/learning_curve50.png" alt="drawing" width="500"/></p>

#### Agent after 700 episodes
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/MountainCarContinuous/results/openaigym.video.25.37024.video000000.gif" width="400"/></p>



## Acrobot-v1
The acrobot system includes two joints and two links, where the joint between the two links is actuated. Initially, the links are hanging downwards, and the goal is to swing the end of the lower link up to a given height. A reward of -1 is provided for every timestep until the goal is reached or 500 timesteps have passed.


### Solution using Double Dueling DQN with Prioritized Experience Replay


<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/Acrobot/results/learning_curve100.png" alt="drawing" width="500"/></p>

#### Agent after 10000 episodes
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/Acrobot/results/openaigym.video.38.37714.video000000.gif" width="400"/></p>


## Pendulum-v0
The inverted pendulum swingup problem is a classic problem in the control literature. In this version of the problem, the pendulum starts in a random position, and the goal is to swing it up so it stays upright.er link up to a given height. A reward of -1 is provided for every timestep until the goal is reached or 500 timesteps have passed.


### Solution using Deep Deterministic Policy Gradient (DDPG) with Prioritized Experience Replay

<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/Pendulum/results/learning_curve50.png" alt="drawing" width="500"/></p>

#### Agent after 1000 episodes
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/Pendulum/results/openaigym.video.10.6156.video000000.gif" width="400"/></p>

# Box2D
Continuous control tasks in the Box2D simulator.

## BipedalWalker-v3
Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. Applying motor torque costs a small amount of points, more optimal agent will get better score. State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. There's no coordinates in the state vector.


### Solution using Soft Actor-Critic (SAC)

<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Box2D/BipedalWalker/results/learning_curve25.png" alt="drawing" width="500"/></p>

#### Agent after 1250 episodes
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Box2D/BipedalWalker/results/gif2.gif" width="400"/></p>

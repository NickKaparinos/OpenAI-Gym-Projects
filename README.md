# OpenAI-Gym-Projects

# Gym
Gym is an open source Python library for developing and comparing reinforcement learning algorithms by providing a
standard API to communicate between learning algorithms and environments, as well as a standard set of environments 
compliant with that API. Since its release, Gym's API has become the field standard for doing this.

# Gym environments solved:
- [Classic Control](#classic-control)
  * [CartPole-v1](#cartpole-v1)
  * [MountainCar-v0](#mountaincar-v0)
  * [MountainCarContinuous-v0](#mountaincarcontinuous-v0)
  * [Acrobot-v1](#acrobot-v1)
  * [Pendulum-v0](#pendulum-v0)
- [Box2D](#box2d)
  * [LunarLander-v2](#lunarlander-v2)
  * [LunarLanderContinuous-v2](#lunarlandercontinuous-v2)
  * [BipedalWalker-v3](#bipedalwalker-v3)
- [MuJoCo](#mujoco)
  * [InvertedPendulum-v2](#invertedpendulum-v2)
  * [Reacher-v2](#reacher-v2)
  * [Hopper2D-v2](#hopper-v2)
  * [Walker2D-v2](#walker2d-v2)
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


### Solution using Double Dueling Deep Q Learning (DQN) with Prioritized Experience Replay


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


### Solution using Double Dueling Deep Q Learning (DQN) with Prioritized Experience Replay


<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/Acrobot/results/learning_curve100.png" alt="drawing" width="500"/></p>

#### Agent after 10000 episodes
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/Acrobot/results/openaigym.video.38.37714.video000000.gif" width="400"/></p>


## Pendulum-v0
The inverted pendulum swingup problem is a classic problem in the control literature. In this version of the problem, the pendulum starts in a random position, and the goal is to swing it up so it stays upright.er link up to a given height. A reward of -1 is provided for every timestep until the goal is reached or 500 timesteps have passed.


### Solution using Deep Deterministic Policy Gradient (DDPG) with Prioritized Experience Replay

<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/Pendulum/results/learning_curve50.png" alt="drawing" width="500"/></p>

#### Agent after 750 episodes
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/Pendulum/results/openaigym.video.10.6156.video000000.gif" width="400"/></p>

# Box2D
Continuous control tasks in the Box2D simulator.

## LunarLander-v2
Navigate the lander to its landing pad. Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.


### Solution using Double Dueling Deep Q Learning (DQN)
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Box2D/LunarLander/results/learning_curve50.png" alt="drawing" width="500"/></p>

#### Agent after 1500 episodes
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Box2D/LunarLander/results/lunarlander.gif" width="400"/></p>

## LunarLanderContinuous-v2
Navigate the lander to its landing pad. Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Action is two real values vector from -1 to +1. First controls main engine, -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power. Second value -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.


### Solution using Soft Actor-Critic (SAC)
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Box2D/LunarLanderContinuous/results/learning_curve25.png" alt="drawing" width="500"/></p>


#### Agent after 1000 episodes
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Box2D/LunarLanderContinuous/results/gif.gif" width="400"/></p>


## BipedalWalker-v3
Train a bipedal robot to walk. Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. Applying motor torque costs a small amount of points, more optimal agent will get better score. State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. There's no coordinates in the state vector.


### Solution using Soft Actor-Critic (SAC)

<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Box2D/BipedalWalker/results/learning_curve25.png" alt="drawing" width="500"/></p>

#### Agent after 1250 episodes
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Box2D/BipedalWalker/results/gif2.gif" width="400"/></p>


# MuJoCo
Continuous control tasks, running in a fast physics simulator.

## InvertedPendulum-v2
An inverted pendulum that needs to be balanced by a cart. The agent gets a reward for every timestep that the pendulum has not fallen off the cart, with a maximum reward of +1000.

### Proximal Policy Optimisation (PPO)

<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/MuJoCo/InvertedPendulum/results/learning_curve50.png" alt="drawing" width="500"/></p>

#### Agent after 1400 episodes
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/MuJoCo/InvertedPendulum/results/inverted_pendulum.gif" width="400"/></p>

#### 100 episode performance evaluation
Reward: 920.71± 224.04

## Reacher-v2
A 2D robot trying to reach a randomly located target. The robot gets a negative reward the furthest away it is from the target location.

### Solution using Soft Actor-Critic (SAC)
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/MuJoCo/Reacher/results/learning_curve50.png" alt="drawing" width="500"/></p>

#### Agent after 3000 episodes
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/MuJoCo/Reacher/results/reacher.gif" width="400"/></p>

#### 100 episode performance evaluation
Reward: -4.75 ± 1.67

## Hopper-v2
A 2D robot that learns to hop. The agent gets a positive reward the furthest it travels.

### Solution using Soft Actor-Critic (SAC)
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/MuJoCo/Hopper/results/learning_curve50.png" alt="drawing" width="500"/></p>

#### Agent after 1450 episodes
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/MuJoCo/Hopper/results/hopper.gif" width="400"/></p>

#### 100 episode performance evaluation
Reward: 3625.53 ± 9.00

## Walker2d-v2
A 2D robot that learns to walk. The agent gets a positive reward the furthest it travels.

### Solution using Soft Actor-Critic  (SAC)
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/MuJoCo/Walker2d/results/learning_curve100.png" alt="drawing" width="500"/></p>

#### Agent after 3250 episodes
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/MuJoCo/Walker2d/results/walker2d.gif" width="400"/></p>

#### 100 episode performance evaluation
Reward: 5317.38 ± 15.86

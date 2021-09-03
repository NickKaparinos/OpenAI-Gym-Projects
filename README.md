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

<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/CartPole/images/learning_curve.png" alt="drawing" width="500"/></p>

#### Agent after 500000 training steps
<p align="center"><img src="https://github.com/NickKaparinos/OpenAI-Gym-Projects/blob/master/Classic%20Control/CartPole/images/ppo.gif" width="400"/></p>

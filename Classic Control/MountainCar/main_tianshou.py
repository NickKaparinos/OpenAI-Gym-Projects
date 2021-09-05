"""
Open AI Gym MountainCar-v0
Nick Kaparinos
2021
"""

import time

from gym import wrappers
import gym
import tianshou as ts
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger, LazyLogger
import torch
from torch import nn
from tianshou.utils.net.common import Net

from utilities import *

if __name__ == '__main__':
    start = time.perf_counter()
    # env_id = "MountainCar-v0"
    env_id = "CartPole-v1"
    seed = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    # Logging directory
    model_name = 'Tianshou_DD_DQN'
    log_dir = 'logs/' + model_name + '_' + str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime())) + '/'
    writer = SummaryWriter(log_dir=log_dir)
    logger = TensorboardLogger(writer, train_interval=1, update_interval=1)

    # Environment
    env = gym.make(env_id)
    train_envs = ts.env.DummyVectorEnv([lambda: gym.make(env_id) for _ in range(1)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(env_id) for _ in range(1)])

    # Neural network
    Q_param = {"hidden_sizes": [128, 128]}
    V_param = {"hidden_sizes": [128, 128]}
    learning_rate = 1e-3
    model_hyperparameters = {'Q_param': Q_param['hidden_sizes'], 'V_param': V_param['hidden_sizes'],
                             'learning_rate': learning_rate}
    save_dict_to_file(model_hyperparameters, path=log_dir, txt_name='model_hyperparameters')
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape, dueling_param=(Q_param, V_param), device=device).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Policy
    policy_hyperparameters = {'discount_factor': 0.99, 'estimation_step': 3, 'target_update_freq': 100,
                              'is_double': True}
    save_dict_to_file(policy_hyperparameters, path=log_dir, txt_name='policy_hyperparameters')
    policy = ts.policy.DQNPolicy(net, optim, **policy_hyperparameters)

    # Collectors
    train_collector = ts.data.Collector(policy, train_envs,
                                        ts.data.PrioritizedVectorReplayBuffer(20000, 100, alpha=0.1, beta=0.1,
                                                                              device=device),
                                        exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

    # Training
    result = ts.trainer.offpolicy_trainer(policy, train_collector, test_collector, max_epoch=1, step_per_epoch=20000,
                                          step_per_collect=10, update_per_step=0.1, episode_per_test=100, batch_size=64,
                                          train_fn=lambda epoch, env_step: policy.set_eps(0.1),
                                          test_fn=lambda epoch, env_step: policy.set_eps(0.05),
                                          stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
                                          logger=logger)
    print(f'Finished training! Use {result["duration"]}')

    # Learning Curve
    learning_curve_tianshou(log_dir=log_dir, window=10)

    # Video
    env = wrappers.Monitor(env=gym.make(env_id), directory=log_dir + '/video', force=True)
    # test_envs = ts.env.DummyVectorEnv([lambda: gym.make(env_id) for _ in range(1)])

    # Video
    policy.eval()
    policy.set_eps(0.01)
    collector = ts.data.Collector(policy, env, exploration_noise=True)
    collector.collect(n_episode=1, render=1 / 60)

    # Execution Time
    end = time.perf_counter()  # tensorboard --logdir './Classic Control/MountainCar/logs'
    print(f"\nExecution time = {end - start:.2f} second(s)")

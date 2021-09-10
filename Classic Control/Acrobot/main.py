"""
Open AI Gym Acrobot-v1
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
    env_id = "Acrobot-v1"
    seed = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Logging directory
    model_name = 'Tianshou_ER_DD_DQN'
    log_dir = 'logs/' + model_name + '_' + str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime())) + '/'
    writer = SummaryWriter(log_dir=log_dir)
    logger = TensorboardLogger(writer, train_interval=1, update_interval=1)

    # Environment
    env = gym.make(env_id)
    env.seed(seed=seed)
    train_envs = ts.env.DummyVectorEnv([lambda: gym.make(env_id) for _ in range(1)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(env_id) for _ in range(1)])

    # Neural network
    Q_param = {"hidden_sizes": [128, 128]}
    V_param = {"hidden_sizes": [128, 128]}
    learning_rate = 1e-3
    model_hyperparameters = {'Q_param': Q_param['hidden_sizes'], 'V_param': V_param['hidden_sizes'],
                             'learning_rate': learning_rate}
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape, dueling_param=(Q_param, V_param), device=device).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Policy
    policy_hyperparameters = {'discount_factor': 0.99, 'estimation_step': 1, 'target_update_freq': 10,
                              'is_double': True}
    policy = ts.policy.DQNPolicy(net, optim, **policy_hyperparameters)

    # Collectors
    prioritized_buffer_hyperparameters = {'total_size': 1_000_000, 'buffer_num': 1, 'alpha': 0.7, 'beta': 0.5}
    train_collector = ts.data.Collector(policy, train_envs,
                                        ts.data.PrioritizedVectorReplayBuffer(**prioritized_buffer_hyperparameters,
                                                                              device=device),
                                        exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

    # Epsilon schedule
    def build_epsilon_schedule(max_epsilon=0.5, min_epsilon=0.0, num_episodes_decay=10000):
        def custom_epsilon_schedule(epoch, env_step):
            decay_step = (max_epsilon - min_epsilon) / num_episodes_decay
            current_epsilon = max_epsilon - env_step * decay_step
            if current_epsilon < 0.0:
                current_epsilon = 0
            policy.set_eps(current_epsilon)

        return custom_epsilon_schedule

    # Test function
    def build_test_fn(num_episodes):
        def custom_test_fn(epoch, env_step):
            print(f"Epoch = {epoch}")

            # Save agent
            torch.save(policy.state_dict(), log_dir + f'dqn_epoch{epoch}.pth')

            # No exploration
            policy.set_eps(0.00)

            # Record agents performance in video
            for episode in range(num_episodes):
                env = ts.env.DummyVectorEnv([lambda: wrappers.Monitor(env=gym.make(env_id),
                                                                      directory=log_dir + '/videos/epoch_' + str(
                                                                          epoch) + '/video' + str(episode), force=False)
                                             for _ in range(1)])

                # Video
                policy.eval()
                policy.set_eps(0.00)
                collector = ts.data.Collector(policy, env, exploration_noise=True)
                collector.collect(n_episode=1, render=1 / 60)

        return custom_test_fn


    # Training
    trainer_hyperparameters = {'max_epoch': 5, 'step_per_epoch': 80_000, 'step_per_collect': 5,
                               'episode_per_test': 10,
                               'batch_size': 64}
    epsilon_schedule_hyperparameters = {'max_epsilon': 0.7, 'min_epsilon': 0.0,
                                        'num_episodes_decay': int(trainer_hyperparameters['step_per_epoch'] * 0.4)}
    all_hypeparameters = model_hyperparameters | policy_hyperparameters | prioritized_buffer_hyperparameters | trainer_hyperparameters | epsilon_schedule_hyperparameters
    all_hypeparameters['seed'] = seed
    save_dict_to_file(all_hypeparameters, path=log_dir)
    result = ts.trainer.offpolicy_trainer(policy, train_collector, test_collector, **trainer_hyperparameters,
                                          train_fn=build_epsilon_schedule(**epsilon_schedule_hyperparameters),
                                          test_fn=build_test_fn(num_episodes=5),
                                          stop_fn=None,
                                          logger=logger)
    print(f'Finished training! Use {result["duration"]}')

    # Learning Curve
    learning_curve_tianshou(log_dir=log_dir, window=50)

    # Record Episode Video
    num_episodes = 10
    for i in range(num_episodes):
        env = ts.env.DummyVectorEnv([lambda: wrappers.Monitor(env=gym.make(env_id),
                                                              directory=log_dir + '/videos/final_agent/video' + str(i),
                                                              force=False) for _ in range(1)])

        # Video
        policy.eval()
        policy.set_eps(0.00)
        collector = ts.data.Collector(policy, env, exploration_noise=True)
        collector.collect(n_episode=1, render=1 / 60)

    # Save policy
    torch.save(policy.state_dict(), log_dir + 'dqn.pth')

    # Execution Time
    end = time.perf_counter()  # tensorboard --logdir './Classic Control/Acrobot/logs'
    print(f"\nExecution time = {end - start:.2f} second(s)")

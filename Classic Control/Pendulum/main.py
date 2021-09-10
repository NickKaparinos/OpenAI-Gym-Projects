"""
Open AI Gym Pendulum
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
from tianshou.utils import TensorboardLogger
from tianshou.exploration import GaussianNoise, OUNoise
from tianshou.policy import DDPGPolicy
import torch
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from utilities import *

if __name__ == '__main__':
    start = time.perf_counter()
    env_id = "Pendulum-v0"
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Logging directory
    model_name = 'Tianshou_DDPG'
    log_dir = 'logs/' + model_name + '_' + str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime())) + '/'
    writer = SummaryWriter(log_dir=log_dir)
    logger = TensorboardLogger(writer, train_interval=1, update_interval=1)

    # Environment
    env = gym.make(env_id)
    env.seed(seed=seed)

    train_envs = ts.env.DummyVectorEnv([lambda: gym.make(env_id) for _ in range(1)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(env_id) for _ in range(1)])
    train_envs.seed(seed)
    test_envs.seed(seed)

    # Neural networks and policy
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    model_hyperparameters = {'hidden_sizes': [128, 128], 'learning_rate': 1e-3, 'estimation_step': 1}

    net_a = Net(state_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], device=device)
    actor = Actor(net_a, action_shape, max_action=max_action, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=model_hyperparameters['learning_rate'])
    net_c = Net(state_shape, action_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], concat=True,
                device=device)
    critic = Critic(net_c, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=model_hyperparameters['learning_rate'])
    policy = DDPGPolicy(actor, actor_optim, critic, critic_optim,
                        exploration_noise=GaussianNoise(sigma=0.5 * max_action),
                        estimation_step=model_hyperparameters['estimation_step'],
                        action_space=env.action_space)

    # Collectors
    prioritized_buffer_hyperparameters = {'total_size': 1_000_000, 'buffer_num': 1, 'alpha': 0.7, 'beta': 0.5}
    train_collector = ts.data.Collector(policy, train_envs,
                                        ts.data.PrioritizedVectorReplayBuffer(**prioritized_buffer_hyperparameters,
                                                                              device=device),
                                        exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)


    # Sigma schedule
    def build_sigma_schedule(max_sigma=0.5, min_sigma=0.0, steps_per_epoch=50000, decay_time_steps=10000):
        def custom_sigma_schedule(epoch, env_step):
            decay_per_step = (max_sigma - min_sigma) / decay_time_steps
            step_number = (epoch - 1) * steps_per_epoch + env_step

            current_sigma = max_sigma - step_number * decay_per_step
            if current_sigma < 0.0:
                current_sigma = 0.0
            policy._noise = GaussianNoise(sigma=current_sigma * max_action)

        return custom_sigma_schedule


    # Test function
    def build_test_fn(num_episodes):
        def custom_test_fn(epoch, env_step):
            print(f"Epoch = {epoch}")

            # Save agent
            torch.save(policy.state_dict(), log_dir + f'dqn_epoch{epoch}.pth')

            # Record agents performance in video
            for episode in range(num_episodes):
                env = ts.env.DummyVectorEnv(
                    [lambda: wrappers.Monitor(env=wrappers.TimeLimit(env=gym.make(env_id), max_episode_steps=200),
                                              directory=log_dir + '/videos/epoch_' + str(
                                                  epoch) + '/video' + str(episode), force=False)
                     for _ in range(1)])

                # Video
                policy.eval()
                collector = ts.data.Collector(policy, env, exploration_noise=True)
                collector.collect(n_episode=1, render=1 / 60)

        return custom_test_fn


    # Training
    trainer_hyperparameters = {'max_epoch': 2, 'step_per_epoch': 75_000, 'step_per_collect': 5,
                               'episode_per_test': 10,
                               'batch_size': 64}
    all_hypeparameters = model_hyperparameters | prioritized_buffer_hyperparameters | trainer_hyperparameters
    all_hypeparameters['seed'] = seed
    save_dict_to_file(all_hypeparameters, path=log_dir)
    decay_steps = int(trainer_hyperparameters['max_epoch'] * trainer_hyperparameters['step_per_epoch'] * 0.2)
    result = ts.trainer.offpolicy_trainer(policy, train_collector, test_collector, **trainer_hyperparameters,
                                          train_fn=build_sigma_schedule(max_sigma=0.2, min_sigma=0.0,
                                                                        steps_per_epoch=trainer_hyperparameters[
                                                                            'step_per_epoch'],
                                                                        decay_time_steps=decay_steps),
                                          test_fn=build_test_fn(num_episodes=2), stop_fn=None, logger=logger)
    print(f'Finished training! Use {result["duration"]}')

    # Learning Curve
    learning_curve_tianshou(log_dir=log_dir, window=50)

    # Record Episode Video
    num_episodes = 10
    for i in range(num_episodes):
        env = ts.env.DummyVectorEnv(
            [lambda: wrappers.Monitor(env=wrappers.TimeLimit(env=gym.make(env_id), max_episode_steps=200),
                                      directory=log_dir + '/videos/final_agent/video' + str(i),
                                      force=False) for _ in range(1)])

        # Video
        policy.eval()
        collector = ts.data.Collector(policy, env, exploration_noise=False)
        collector.collect(n_episode=1, render=1 / 60)

    # Save policy
    torch.save(policy.state_dict(), log_dir + model_name + '.pth')

    # Execution Time
    end = time.perf_counter()  # tensorboard --logdir './Classic Control/Pendulum/logs'
    print(f"\nExecution time = {end - start:.2f} second(s)")

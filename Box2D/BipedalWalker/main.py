"""
Open AI Gym BipedalWalker-v3
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
from tianshou.policy import SACPolicy
import torch
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Critic, ActorProb
from utilities import *

if __name__ == '__main__':
    start = time.perf_counter()
    env_id = "BipedalWalker-v3"
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_pretrained_model = False
    model_path = ''

    # Logging directory
    model_name = 'Tianshou_SAC'
    log_dir = 'logs/' + model_name + '_' + str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime())) + '/'
    writer = SummaryWriter(log_dir=log_dir)
    logger = TensorboardLogger(writer, train_interval=1, update_interval=1)

    # Environment
    env = gym.make(env_id)
    env.seed(seed=seed)

    train_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(env_id) for _ in range(1)])
    test_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(env_id) for _ in range(1)])
    train_envs.seed(seed)
    test_envs.seed(seed)

    # Neural networks and policy
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    model_hyperparameters = {'hidden_sizes': [256, 256], 'learning_rate': 1e-3, 'estimation_step': 4}

    # Actor
    net_a = Net(state_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], device=device)
    actor = ActorProb(net_a, action_shape, max_action=max_action, device=device, unbounded=True).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=model_hyperparameters['learning_rate'])

    # Critics
    net_c1 = Net(state_shape, action_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], concat=True,
                 device=device)
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=model_hyperparameters['learning_rate'])
    net_c2 = Net(state_shape, action_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], concat=True,
                 device=device)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=model_hyperparameters['learning_rate'])

    # Alpha
    target_entropy = -np.prod(env.action_space.shape)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_lr = 1e-5
    alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
    alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
                       exploration_noise=GaussianNoise(sigma=2 * max_action),
                       estimation_step=model_hyperparameters['estimation_step'],
                       action_space=env.action_space, alpha=alpha)

    # Collectors
    use_prioritised_replay_buffer = False
    prioritized_buffer_hyperparameters = {'total_size': 1_000_000, 'buffer_num': 1, 'alpha': 0.4, 'beta': 0.5}
    if use_prioritised_replay_buffer:
        train_collector = ts.data.Collector(policy, train_envs,
                                            ts.data.PrioritizedVectorReplayBuffer(**prioritized_buffer_hyperparameters),
                                            exploration_noise=True)
    else:
        train_collector = ts.data.Collector(policy, train_envs,
                                            ts.data.ReplayBuffer(size=prioritized_buffer_hyperparameters['total_size']),
                                            exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    # Sigma schedule
    def build_sigma_schedule(max_sigma=0.5, min_sigma=0.0, steps_per_epoch=50_000, decay_time_steps=10_000):
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
            torch.save(policy.state_dict(), log_dir + model_name + f'_epoch{epoch}.pth')

            # Record agents performance in video
            for episode in range(num_episodes):
                env = ts.env.DummyVectorEnv(
                    [lambda: wrappers.Monitor(env=gym.make(env_id), directory=log_dir + '/videos/epoch_' + str(
                        epoch) + '/video' + str(episode), force=False) for _ in range(1)])

                # Video
                policy.eval()
                collector = ts.data.Collector(policy, env, exploration_noise=True)
                collector.collect(n_episode=1, render=1 / 60)

        return custom_test_fn


    # Training
    trainer_hyperparameters = {'max_epoch': 6, 'step_per_epoch': 200_000, 'step_per_collect': 10,
                               'episode_per_test': 10,
                               'batch_size': 128}
    decay_steps = int(trainer_hyperparameters['max_epoch'] * trainer_hyperparameters['step_per_epoch'] * 0.05)
    build_sigma_hyperparameters = {'max_sigma': 0.6, 'min_sigma': 0.0, 'decay_time_steps': decay_steps}
    all_hypeparameters = dict(model_hyperparameters, **trainer_hyperparameters, **prioritized_buffer_hyperparameters)
    all_hypeparameters['seed'] = seed
    all_hypeparameters['use_prioritised_replay_buffer'] = use_prioritised_replay_buffer
    all_hypeparameters['alpha_lr'] = alpha_lr
    if load_pretrained_model:
        policy.load_state_dict(torch.load(model_path))
        all_hypeparameters['load_pretrained_model'] = load_pretrained_model
        all_hypeparameters['model_path'] = model_path
    save_dict_to_file(all_hypeparameters, path=log_dir)

    result = ts.trainer.offpolicy_trainer(policy, train_collector, test_collector, **trainer_hyperparameters,
                                          train_fn=build_sigma_schedule(**build_sigma_hyperparameters,
                                                                        steps_per_epoch=trainer_hyperparameters[
                                                                            'step_per_epoch']),
                                          test_fn=build_test_fn(num_episodes=5), stop_fn=None, logger=logger)
    print(f'Finished training! Use {result["duration"]}')

    # Learning Curve
    learning_curve_tianshou(log_dir=log_dir, window=100)

    # Record Episode Video
    num_episodes = 10
    for i in range(num_episodes):
        env = ts.env.DummyVectorEnv(
            [lambda: wrappers.Monitor(env=gym.make(env_id), directory=log_dir + '/videos/final_agent/video' + str(i),
                                      force=False) for _ in range(1)])

        # Video
        policy.eval()
        collector = ts.data.Collector(policy, env, exploration_noise=False)
        collector.collect(n_episode=1, render=1 / 60)

    # Save policy
    torch.save(policy.state_dict(), log_dir + model_name + '.pth')

    # Execution Time
    end = time.perf_counter()  # tensorboard --logdir './Box2D/BipedalWalker/logs'
    print(f"\nExecution time = {end - start:.2f} second(s)")

"""
Open AI Gym Pong
Nick Kaparinos
2022
"""

import time
import envpool
import tianshou as ts
import torch.optim
from tianshou.data import PrioritizedVectorReplayBuffer as PVRB, VectorReplayBuffer as VRB
from tianshou.policy import DQNPolicy
from tianshou.policy.modelbased.icm import ICMPolicy
from tianshou.utils import WandbLogger
from tianshou.utils.net.discrete import IntrinsicCuriosityModule
import wandb
from torch.utils.tensorboard import SummaryWriter
from atari_network import DQN
from utilities import *


def main():
    """ Train and evaluate RL Agent on Pong Atari environment """
    start = time.perf_counter()
    seed = 0
    set_all_seeds(seed=seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # RL Environment
    env_id = 'Pong-v5'

    # Config
    num_envs, num_videos, num_final_episodes = 5, 4, 6
    continue_from_previous_run, latest_run_epoch = True, 1
    previous_runs = ['DQN_22_May_2022_12_29_24']
    trainer_hyperparameters = {'max_epoch': 2, 'step_per_epoch': 1_000, 'step_per_collect': 100,
                               'episode_per_test': 100, 'batch_size': 128, 'update_per_step': 0.1}
    policy_hyperparameters = {'discount_factor': 0.99, 'estimation_step': 1, 'target_update_freq': 20,
                              'is_double': True}
    epsilon_schedule_hyperparameters = {'max_epsilon': 0.0, 'min_epsilon': 0.0, 'num_episodes_decay': int(
        trainer_hyperparameters['step_per_epoch'] * trainer_hyperparameters['max_epoch'] * 0.9)}
    replay_buffer_hyperparameters = {'total_size': 100_000, 'buffer_num': num_envs, 'alpha': 0.7, 'beta': 0.5,
                                     'ignore_obs_next': True, 'save_only_last_obs': True, 'stack_num': 4}
    icm_hyperparameters = {'use_icm': False, 'icm_hidden_sizes': [256], 'icm_lr': 1e-4, 'icm_lr_scale': 0.1,
                           'icm_reward_scale': 0.1, 'icm_forward_loss_weight': 0.2}
    misq_dict = {'learning_rate': 1e-4, 'seed': seed, 'use_prioritised_replay_buffer': True,
                 'optimizer': 'Adam', 'continue_from_previous_run': continue_from_previous_run, 'algorithm': 'DQN',
                 'latest_run_epoch': latest_run_epoch, 'previous_runs': previous_runs}
    config = dict(policy_hyperparameters, **trainer_hyperparameters, **epsilon_schedule_hyperparameters,
                  **replay_buffer_hyperparameters, **icm_hyperparameters, **misq_dict)

    # Logging
    model_name = f'{config["algorithm"]}_' + str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    log_dir = f'logs/{model_name}/'
    makedirs(log_dir, exist_ok=True)
    project = 'Gym-' + env_id[:-3]
    logger = WandbLogger(train_interval=1, save_interval=1, project=project, entity='nickkaparinos', name=model_name,
                         run_id=model_name, config=config)  # type: ignore
    logger.load(SummaryWriter(log_dir))

    # Environment
    train_envs = env = envpool.make_gym(env_id, num_envs=num_envs, seed=seed, episodic_life=True, reward_clip=True,
                                        stack_num=replay_buffer_hyperparameters['stack_num'])
    test_envs = envpool.make_gym(env_id, num_envs=num_envs, seed=seed, episodic_life=False, reward_clip=True,
                                 stack_num=replay_buffer_hyperparameters['stack_num'])

    # Neural network
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = DQN(*state_shape, action_shape, device).to(device)  # type: ignore

    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD
    else:
        raise ValueError(f'Optimizer: {config["optimizer"]} not supported')
    optim = optimizer(net.parameters(), lr=config['learning_rate'])

    # Policy
    policy = DQNPolicy(net, optim, **policy_hyperparameters)

    # Intrinsic Motivation Module
    if config['use_icm']:
        feature_net = DQN(*state_shape, action_shape, device, features_only=True)
        action_dim = np.prod(action_shape)
        feature_dim = feature_net.output_dim
        icm_net = IntrinsicCuriosityModule(feature_net.net, feature_dim, action_dim,
                                           hidden_sizes=config['icm_hidden_sizes'], device=device)  # noqa
        icm_optim = torch.optim.Adam(icm_net.parameters(), lr=config['icm_lr'])
        policy = ICMPolicy(policy, icm_net, icm_optim, config['icm_lr_scale'], config['icm_reward_scale'],
                           config['icm_forward_loss_weight']).to(device)

    # Collectors
    if config['use_prioritised_replay_buffer']:
        train_collector = ts.data.Collector(policy, train_envs, PVRB(**replay_buffer_hyperparameters),
                                            exploration_noise=True)
    else:
        train_collector = ts.data.Collector(policy, train_envs,
                                            VRB(total_size=replay_buffer_hyperparameters['total_size'] * num_envs,
                                                buffer_num=num_envs), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

    # Load previous run
    if continue_from_previous_run:
        load_previous_run(previous_runs, latest_run_epoch, policy, train_collector)

    # Training
    policy.set_eps(epsilon_schedule_hyperparameters['max_epsilon'])
    save_dict_to_txt(config, path=log_dir, txt_name='config')
    train_fn = build_epsilon_schedule(policy=policy, **epsilon_schedule_hyperparameters)
    test_fn = build_test_fn(policy, optim, log_dir, model_name, train_collector, True, state_shape[1:],
                            config['stack_num'], env_id, num_videos)
    result = ts.trainer.offpolicy_trainer(policy, train_collector, test_collector, **trainer_hyperparameters,
                                          train_fn=train_fn, test_fn=test_fn, stop_fn=None, logger=logger)
    print(f'Finished training! Duration {result["duration"]}')

    # Learning curve
    windows = [10, 25, 50]
    make_learning_curve(project, model_name, previous_runs, log_dir, windows, continue_from_previous_run)

    # Visualize agent`s policy
    epoch = config['max_epoch'] + 1
    collect_and_visualize_episodes(policy, log_dir, epoch, env_id, num_final_episodes, state_shape[1:],
                                   config['stack_num'])

    # Execution Time
    wandb.finish()
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")


if __name__ == '__main__':
    main()

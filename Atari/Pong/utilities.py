"""
Open AI Gym Pong
Nick Kaparinos
2022
"""

import gym
import random
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tianshou as ts
from os import makedirs
from tianshou.data import PrioritizedVectorReplayBuffer as PVRB
import record
import envpool


def make_learning_curve(project, model_name, previous_runs, log_dir, windows, continue_from_previous_run):
    """ Make learning curve and log to wandb """
    api = wandb.Api()  # noqa
    run = api.run(f'nickkaparinos/{project}/{model_name}')
    history = run.scan_history()
    episode_rewards = get_episode_rewards(history)

    # Download previous runs data
    for previous_run in reversed(previous_runs):
        if continue_from_previous_run:
            run = api.run('nickkaparinos' + '/' + project + '/' + previous_run)
            history = run.scan_history()
            episode_rewards_temp = get_episode_rewards(history)
            episode_rewards = pd.concat([episode_rewards_temp, episode_rewards], axis=0).reset_index(drop=True)

    # Learning Curve
    for window in windows:
        plot_learning_curve(episode_rewards=episode_rewards, log_dir=log_dir, window=window)


def plot_learning_curve(episode_rewards, log_dir, window=10):
    """ Plot learning curve given episode reward list """
    # Calculate rolling window metrics
    rolling_average = episode_rewards.rolling(window=window, min_periods=1).mean().dropna()
    rolling_max = episode_rewards.rolling(window=window, min_periods=1).max().dropna()
    rolling_min = episode_rewards.rolling(window=window, min_periods=1).min().dropna()

    # Change column name
    rolling_average.columns = ['Average Reward']
    rolling_max.columns = ['Max Reward']
    rolling_min.columns = ['Min Reward']
    rolling_data = pd.concat([rolling_average, rolling_max, rolling_min], axis=1)

    # Plot
    sns.set()
    plt.figure(0, dpi=200)
    plt.clf()
    ax = sns.lineplot(data=rolling_data)
    ax.fill_between(rolling_average.index, rolling_min.iloc[:, 0], rolling_max.iloc[:, 0], alpha=0.2)
    ax.set_title('Learning Curve', fontsize=16)
    ax.set_ylabel('Reward')
    ax.set_xlabel('Episodes')

    # Save figure
    img_path = f'{log_dir}learning_curve{window}.png'
    plt.savefig(img_path, dpi=200)

    # Log figure
    image = plt.imread(img_path)
    wandb.log({f'Learning_Curve_{window}': [wandb.Image(image, caption="Learning_curve")]})  # type: ignore


def get_episode_rewards(history) -> pd.DataFrame:
    """ Get episode rewards from wandb scan history """
    episode_rewards_temp = []
    for i in history:
        if 'train/reward' in i.keys():
            episode_rewards_temp.append(i['train/reward'])
    return pd.DataFrame(data=episode_rewards_temp, columns=['Rewards'])


def build_test_fn(policy, optim, log_dir, model_name, train_collector, save_train_buffer, obs_shape, stack_num, env_id,
                  num_episodes):
    """ Build custom test function for maze world environment """

    def custom_test_fn(epoch, env_step):
        # Save agent
        print(f"Epoch = {epoch}")
        torch.save({'model': policy.state_dict(), 'optim': optim.state_dict()},
                   log_dir + model_name + f'_epoch{epoch}.pth')
        if save_train_buffer:
            train_collector.buffer.save_hdf5(f'{log_dir}/epoch{epoch}_train_buffer.hdf5')

        # Record agent`s performance in video
        policy.eval()
        test_env = envpool.make_gym(env_id, num_envs=1, seed=0, episodic_life=False, reward_clip=True, stack_num=4,
                                    gray_scale=False, img_height=160, img_width=160)
        collector = ts.data.Collector(policy, test_env, exploration_noise=True)
        record.collect_and_record(collector, n_episode=num_episodes // 2, obs_shape=obs_shape, stack_num=stack_num,
                                  log_dir=log_dir, epoch=epoch, starting_episode=0)

        collector = ts.data.Collector(policy, test_env, exploration_noise=False)
        record.collect_and_record(collector, n_episode=num_episodes // 2, obs_shape=obs_shape, stack_num=stack_num,
                                  log_dir=log_dir, epoch=epoch, starting_episode=num_episodes // 2)

    return custom_test_fn


def build_epsilon_schedule(policy, max_epsilon=0.5, min_epsilon=0.0, num_episodes_decay=10000):
    """ Build epsilon schedule function """

    def custom_epsilon_schedule(epoch, env_step):
        decay_step = (max_epsilon - min_epsilon) / num_episodes_decay
        current_epsilon = max_epsilon - env_step * decay_step
        if current_epsilon < min_epsilon:
            current_epsilon = min_epsilon
        policy.set_eps(current_epsilon)
        wandb.log({"train/env_step": env_step, 'epsilon': current_epsilon})  # type: ignore

    return custom_epsilon_schedule


def load_previous_run(previous_runs, latest_run_epoch, policy, train_collector):
    """ Load model, optimizers and buffer from previous run"""
    assert latest_run_epoch > 0
    latest_run = previous_runs[-1]
    checkpoint = torch.load(f'logs/{latest_run}/{latest_run}_epoch{latest_run_epoch}.pth')
    policy.load_state_dict(checkpoint['model'])
    policy.optim.load_state_dict(checkpoint['optim'])
    train_collector.buffer = PVRB.load_hdf5(
        f'logs/{latest_run}/epoch{latest_run_epoch}_train_buffer.hdf5')


def save_dict_to_txt(dictionary, path, txt_name='hyperparameter_dict'):
    """ Save dictionary as txt file """
    with open(f'{path}/{txt_name}.txt', 'w') as f:
        f.write(str(dictionary))


def save_list_to_file(my_list: list, name: str):
    with open(name, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)


def set_all_seeds(seed):
    """ Set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore


def collect_and_visualize_episodes(policy, log_dir, epoch, env_id, num_episodes, obs_shape, stack_num=4):
    """ Collect num_episodes episodes and visualize agent`s policy """
    policy.eval()
    makedirs(f'{log_dir}epoch-{epoch}', exist_ok=True)

    # Record agent`s performance in video
    policy.eval()
    test_env = envpool.make_gym(env_id, num_envs=1, seed=0, episodic_life=False, reward_clip=True, stack_num=4,
                                gray_scale=False, img_height=160, img_width=160)
    collector = ts.data.Collector(policy, test_env, exploration_noise=False)
    collector.collect_and_record = record.collect_and_record
    collector.collect_and_record(collector, n_episode=num_episodes, obs_shape=obs_shape, stack_num=stack_num,
                                 log_dir=log_dir, epoch=epoch)

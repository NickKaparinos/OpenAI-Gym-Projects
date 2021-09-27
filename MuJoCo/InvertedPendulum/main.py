"""
Open AI Gym InvertedPendulum-v2
Nick Kaparinos
2021
"""

import time

from gym import wrappers
import gym
import tianshou as ts
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.exploration import GaussianNoise, OUNoise
from tianshou.policy import PPOPolicy
import torch
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Critic, ActorProb
from utilities import *

if __name__ == '__main__':
    start = time.perf_counter()
    env_id = "InvertedPendulum-v2"
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_pretrained_model = False
    model_path = ''

    # Logging directory
    model_name = 'Tianshou_PPO'
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
    model_hyperparameters = {'hidden_sizes': [64, 64], 'learning_rate': 1e-3}

    # Actor
    net_a = Net(state_shape, activation=torch.nn.Tanh, hidden_sizes=model_hyperparameters['hidden_sizes'],
                device=device)
    actor = ActorProb(net_a, action_shape, max_action=max_action, device=device, unbounded=True).to(device)

    # Critics
    net_c = Net(state_shape, action_shape, activation=torch.nn.Tanh, hidden_sizes=model_hyperparameters['hidden_sizes'],
                device=device)
    critic = Critic(net_c, device=device).to(device)

    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                             lr=model_hyperparameters['learning_rate'])


    def dist(*logits):
        return Independent(Normal(*logits), 1)


    # decay learning rate to 0 linearly
    max_epoch = 6
    step_per_epoch = 250_000
    step_per_collect = 1024
    starting_lr = 1e-2
    ending_lr = 5e-6
    max_update_num = np.ceil(step_per_epoch / step_per_collect) * max_epoch
    lr_scheduler = LambdaLR(optim,
                            lr_lambda=lambda epoch: starting_lr - starting_lr * (epoch / max_update_num) + ending_lr)

    policy_hyperparametres = {'ent_coef': 0.01, 'reward_normalization': False, 'advantage_normalization': False,
                              'recompute_advantage': True, 'vf_coef': 0.25, 'lr_scheduler': lr_scheduler,
                              'action_bound_method': 'clip', 'dual_clip': None, 'value_clip': False,
                              'max_grad_norm': 0.6}
    policy = PPOPolicy(actor, critic, optim, dist, **policy_hyperparametres, action_space=env.action_space)

    # Collectors
    use_prioritised_replay_buffer = False
    prioritized_buffer_hyperparameters = {'total_size': 2048, 'buffer_num': 1, 'alpha': 0.4, 'beta': 0.5}
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
        decay_per_step = (max_sigma - min_sigma) / decay_time_steps

        def custom_sigma_schedule(epoch, env_step):
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
                env = ts.env.DummyVectorEnv([lambda: gym.make(env_id) for _ in range(1)])

                # Video
                policy.eval()
                collector = ts.data.Collector(policy, env, exploration_noise=False)
                collector.collect_and_record = collect_and_record
                collector.collect_and_record(self=collector, video_dir=log_dir + f'epoch{epoch}/video{episode}/',
                                             n_episode=1, render=1 / 60)

        return custom_test_fn


    # Training
    trainer_hyperparameters = {'max_epoch': max_epoch, 'step_per_epoch': step_per_epoch,
                               'step_per_collect': step_per_collect,
                               'repeat_per_collect': 10,
                               'episode_per_test': 10,
                               'batch_size': 64}
    decay_steps = int(trainer_hyperparameters['max_epoch'] * trainer_hyperparameters['step_per_epoch'] * 0.001)
    build_sigma_hyperparameters = {'max_sigma': 1, 'min_sigma': 0.0, 'decay_time_steps': decay_steps}
    all_hypeparameters = dict(model_hyperparameters, **trainer_hyperparameters, **prioritized_buffer_hyperparameters,
                              **policy_hyperparametres)
    all_hypeparameters['seed'] = seed
    all_hypeparameters['use_prioritised_replay_buffer'] = use_prioritised_replay_buffer
    all_hypeparameters['starting_lr'] = starting_lr
    all_hypeparameters['ending_lr'] = ending_lr
    if load_pretrained_model:
        policy.load_state_dict(torch.load(model_path))
        all_hypeparameters['load_pretrained_model'] = load_pretrained_model
        all_hypeparameters['model_path'] = model_path
    save_dict_to_file(all_hypeparameters, path=log_dir)

    result = ts.trainer.onpolicy_trainer(policy, train_collector, test_collector, **trainer_hyperparameters,
                                         train_fn=build_sigma_schedule(**build_sigma_hyperparameters,
                                                                       steps_per_epoch=trainer_hyperparameters[
                                                                           'step_per_epoch']),
                                         test_fn=build_test_fn(num_episodes=4), stop_fn=None, logger=logger)
    print(f'Finished training! Use {result["duration"]}')

    # Learning Curve
    learning_curve_tianshou(log_dir=log_dir, window=50)

    # Record Episode Video
    num_episodes = 10
    for episode in range(num_episodes):
        # Video
        env = ts.env.DummyVectorEnv([lambda: gym.make(env_id) for _ in range(1)])
        policy.eval()
        collector = ts.data.Collector(policy, env, exploration_noise=False)
        collector.collect_and_record = collect_and_record
        collector.collect_and_record(self=collector, video_dir=log_dir + f'final_agent/video{episode}/', n_episode=1,
                                     render=1 / 60)

    # Save policy
    torch.save(policy.state_dict(), log_dir + model_name + '.pth')

    # Execution Time
    end = time.perf_counter()  # tensorboard --logdir './MuJoCo/InvertedPendulum/logs'
    print(f"\nExecution time = {end - start:.2f} second(s)")

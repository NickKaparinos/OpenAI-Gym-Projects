"""
Open AI Gym Reacher-v2
Nick Kaparinos
2021
"""

import pickle
import gym
import tianshou as ts
from tianshou.exploration import GaussianNoise
from tianshou.policy import DDPGPolicy
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from torch.utils.tensorboard import SummaryWriter
from utilities import *

if __name__ == '__main__':
    start = time.perf_counter()
    env_id = "Reacher-v2"
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_pretrained_model = False
    model_log_dir, model_file, buffer_file = '', '', ''

    # Logging directory
    model_name = 'Tianshou_DDPG'
    log_dir = 'logs/' + model_name + '_' + str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime())) + '/'
    writer = SummaryWriter(log_dir=log_dir)
    logger = TensorboardLogger(writer, train_interval=1, update_interval=1)

    # Environment
    env = gym.make(env_id)
    env.seed(seed=seed)
    max_episode_steps = 250


    def make_env():
        env = gym.make(env_id)
        env._max_episode_steps = max_episode_steps
        return env


    train_envs = ts.env.DummyVectorEnv([lambda: make_env() for _ in range(1)])
    test_envs = ts.env.DummyVectorEnv([lambda: make_env() for _ in range(1)])
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
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)


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
            torch.save({'model': policy.state_dict(), 'actor_optim': actor_optim.state_dict(),
                        'critic_optim': critic_optim.state_dict()},
                       log_dir + model_name + f'_epoch{epoch}.pth')
            pickle.dump(train_collector.buffer, open(log_dir + f'epoch{epoch}_train_buffer.pkl', "wb"))

            # Record agents performance in video with and without exploration
            for episode in range(num_episodes // 2):
                env = ts.env.DummyVectorEnv([lambda: make_env() for _ in range(1)])
                policy.eval()
                collector = ts.data.Collector(policy, env, exploration_noise=True)
                collector.collect_and_record = collect_and_record
                collector.collect_and_record(self=collector, video_dir=log_dir + f'epoch{epoch}/video{episode}/',
                                             n_episode=1, render=1 / 60)

            for episode in range(num_episodes // 2, num_episodes):
                env = ts.env.DummyVectorEnv([lambda: make_env() for _ in range(1)])
                policy.eval()
                collector = ts.data.Collector(policy, env, exploration_noise=False)
                collector.collect_and_record = collect_and_record
                collector.collect_and_record(self=collector, video_dir=log_dir + f'epoch{epoch}/video{episode}/',
                                             n_episode=1, render=1 / 60)

        return custom_test_fn


    # Training
    trainer_hyperparameters = {'max_epoch': 5, 'step_per_epoch': 75_000, 'step_per_collect': 100,
                               'episode_per_test': 100, 'batch_size': 128}
    decay_steps = int(trainer_hyperparameters['max_epoch'] * trainer_hyperparameters['step_per_epoch'] * 1)
    build_sigma_hyperparameters = {'max_sigma': 1, 'min_sigma': 0.1, 'decay_time_steps': decay_steps}
    all_hypeparameters = dict(model_hyperparameters, **prioritized_buffer_hyperparameters, **trainer_hyperparameters,
                              **build_sigma_hyperparameters)
    all_hypeparameters['seed'] = seed
    if load_pretrained_model:
        # Load model, optimisers and buffer
        checkpoint = torch.load(model_log_dir + model_file)
        policy.load_state_dict(checkpoint['model'])
        policy.actor_optim.load_state_dict(checkpoint['actor_optim'])
        policy.critic_optim.load_state_dict(checkpoint['critic_optim'])
        train_collector.buffer = pickle.load(open(model_log_dir + buffer_file, "rb"))
        all_hypeparameters['load_pretrained_model'] = load_pretrained_model
        all_hypeparameters['model_log_dir'] = model_log_dir
        all_hypeparameters['model_file'] = model_file
        all_hypeparameters['buffer_file'] = buffer_file

    all_hypeparameters['decay_steps'] = decay_steps
    save_dict_to_file(all_hypeparameters, path=log_dir)

    result = ts.trainer.offpolicy_trainer(policy, train_collector, test_collector, **trainer_hyperparameters,
                                          train_fn=build_sigma_schedule(**build_sigma_hyperparameters,
                                                                        steps_per_epoch=trainer_hyperparameters[
                                                                            'step_per_epoch']),
                                          test_fn=build_test_fn(num_episodes=8), stop_fn=None, logger=logger)
    print(f'Finished training! Use {result["duration"]}')

    # Learning Curve
    learning_curve_tianshou(log_dir=log_dir, window=25)

    # Record Episode Video
    num_episodes = 10
    for episode in range(num_episodes):
        # Video
        env = ts.env.DummyVectorEnv([lambda: make_env() for _ in range(1)])
        env.spec[0].max_episode_steps = max_episode_steps
        policy.eval()
        collector = ts.data.Collector(policy, env, exploration_noise=False)
        collector.collect_and_record = collect_and_record
        collector.collect_and_record(self=collector, video_dir=log_dir + f'final_agent/video{episode}/', n_episode=1,
                                     render=1 / 60)

    # Execution Time
    end = time.perf_counter()  # tensorboard --logdir './MuJoCo/Reacher/logs'
    print(f"\nExecution time = {end - start:.2f} second(s)")

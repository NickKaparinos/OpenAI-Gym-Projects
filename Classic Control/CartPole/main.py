"""
Open AI Gym CartPole-v1
Nick Kaparinos
2021
"""

import gym
from stable_baselines3 import PPO, A2C, DQN, DDPG, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from gym import wrappers
from utilities import *
import time

if __name__ == '__main__':
    start = time.perf_counter()
    env_id = "CartPole-v1"
    seed = 0
    set_random_seed(seed=seed)

    # Create environment
    env = make_vec_env(env_id, n_envs=1, seed=0, vec_env_cls=DummyVecEnv)

    # Logging directory
    model_name = 'PPO'
    log_dir = 'logs/' + model_name + '_' + str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime())) + '/'

    # Agent
    model_hyperparameters = {'policy': 'MlpPolicy'}
    model = PPO(**model_hyperparameters, env=env, verbose=0, tensorboard_log=log_dir)
    model.name = model_name

    # Logger
    logger = configure(log_dir, ["csv", "tensorboard"])  # "stdout",
    log_steps_callback = LogStepsCallback(log_dir=log_dir)
    tqdm_callback = TqdmCallback()
    model.set_logger(logger)
    save_dict_to_file(model_hyperparameters, path=log_dir)  # log hyperparameters

    # Learn
    model.learn(total_timesteps=1_000, callback=[tqdm_callback, log_steps_callback])

    # Plot learning curve
    learning_curve(log_dir=log_dir)

    # Render episodes and save video
    render_episodes = True
    num_steps = 500
    if render_episodes:
        env = wrappers.Monitor(env=gym.make('CartPole-v1'), directory=log_dir + '/video', force=True)
        obs = env.reset()
        for _ in range(num_steps):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            if done:
                break

    # Execution Time            # tensorboard --logdir './Classic Control/CartPole/logs'
    end = time.perf_counter()  # tensorboard --logdir '.\Classic Control\CartPole\logs'
    print(f"\nExecution time = {end - start:.2f} second(s)")

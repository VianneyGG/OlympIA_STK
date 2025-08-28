import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env   
from stable_baselines3 import PPO
from tqdm import tqdm
import os

training_env = make_vec_env("LunarLander-v2", n_envs = 16)

model = PPO(
    "MlpPolicy",
    env = training_env,
    learning_rate = 3e-4,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 5,
    clip_range = 0.2,
    gamma = 0.99,
    gae_lambda = 0.95,
    ent_coef = 0.01,
    verbose=1
)

model_name = "ppo_lunarlander"
model.learn(total_timesteps=500000)
model.save(model_name)

env.close()

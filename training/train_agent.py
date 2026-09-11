# train_agent.py — FIXED

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import os

import envs.indoor_nav_env

os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("logs/tensorboard", exist_ok=True)

env = make_vec_env(
    "IndoorNavRGB-v0",
    n_envs=1,
    env_kwargs={"render_mode": "rgb_array"}
)

#  FIX: Use MultiInputPolicy — handles both image + goal_info vector
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,       # small entropy bonus keeps exploration alive
    tensorboard_log="./logs/tensorboard/",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Save checkpoints every 100k steps so we don't lose progress
checkpoint_cb = CheckpointCallback(
    save_freq=100_000,
    save_path="./models/checkpoints/",
    name_prefix="ppo_indoor_nav"
)

print("🚀 Training started...")
model.learn(
    total_timesteps=1_500_000,   # 200k was way too few so use at least 1M
    callback=checkpoint_cb,
    progress_bar=True
)

model.save("models/ppo_cnn_indoor_nav")
print("✅ Model saved!")
env.close()
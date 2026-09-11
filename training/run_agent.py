# run_agent.py — with auto graph at the end

import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from stable_baselines3 import PPO
import envs.indoor_nav_env

model = PPO.load("models/ppo_cnn_indoor_nav")
env = gym.make("IndoorNavRGB-v0", render_mode="human")
obs, _ = env.reset()

print("🚀 Running trained agent...")

# ── tracking ──────────────────────────────────────────────────
episode = 1
step = 0
episode_reward = 0
all_rewards = []       # total reward per episode
all_lengths = []       # steps per episode
outcomes = []          # "Goal" / "Collision" / "Timeout"

while episode <= 20:   # run 20 episodes then plot — change this number if you want more/less
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    episode_reward += reward
    step += 1

    print(f"[Ep {episode:2d}] Step: {step:3d} | Action: {action} | Reward: {reward:6.2f}")
    time.sleep(0.03)

    if terminated or truncated:
        all_rewards.append(episode_reward)
        all_lengths.append(step)

        if reward >= 50:
            result = "Goal ✅"
            outcomes.append("Goal")
        elif reward <= -15:
            result = "Collision ❌"
            outcomes.append("Collision")
        else:
            result = "Timeout ⏱"
            outcomes.append("Timeout")

        print(f"  → Episode {episode} done | Total Reward: {episode_reward:.2f} | {result}\n")

        episode += 1
        step = 0
        episode_reward = 0
        obs, _ = env.reset()

env.close()

# ── plot ──────────────────────────────────────────────────────
episodes = list(range(1, len(all_rewards) + 1))

color_map = {"Goal": "green", "Collision": "red", "Timeout": "orange"}
bar_colors = [color_map[o] for o in outcomes]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("PPO Agent — Run Results (Indoor Navigation)", fontsize=14, fontweight="bold")

# Reward per episode
ax1.bar(episodes, all_rewards, color=bar_colors, edgecolor="black", linewidth=0.5)
ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax1.set_title("Total Reward per Episode")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.set_xticks(episodes)

# Legend
from matplotlib.patches import Patch
legend = [Patch(color="green", label="Goal Reached"),
          Patch(color="red",   label="Collision"),
          Patch(color="orange",label="Timeout")]
ax1.legend(handles=legend)

# Episode length
ax2.plot(episodes, all_lengths, marker="o", color="#1A56A0", linewidth=2, markersize=5)
ax2.set_title("Steps Taken per Episode")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Steps")
ax2.set_xticks(episodes)

plt.tight_layout()
plt.savefig("agent_run_results.png", dpi=150, bbox_inches="tight")
print("\n📊 Graph saved as agent_run_results.png")
plt.show()
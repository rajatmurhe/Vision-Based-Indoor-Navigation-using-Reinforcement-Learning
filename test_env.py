import os
import time
from envs.indoor_nav_env import IndoorNavEnv

print("RUNNING FILE:", os.path.abspath(__file__))

env = IndoorNavEnv(render_mode="human")
env.reset()

print("PYBULLET WINDOW SHOULD BE OPEN NOW")
print("DO NOT CLOSE TERMINAL")

while True:
    time.sleep(1)

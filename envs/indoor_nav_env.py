# indoor_nav_env.py 

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium.envs.registration import register
import random


class IndoorNavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.action_space = spaces.Discrete(4)  # 0=F, 1=L, 2=R, 3=STOP

        # ✅ FIX 1: Dict obs — image + goal direction vector
        # The agent now has a "compass" to know where the goal is
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "goal_info": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            # goal_info = [norm_distance, cos(rel_angle), sin(rel_angle), step_progress]
        })

        self.step_count = 0
        self.max_steps = 300
        self.obstacle_ids = []
        self.goal_position = None

        self._build_environment()

    def _random_position(self):
        return np.array([
            random.uniform(-3.5, 3.5),
            random.uniform(-3.5, 3.5)
        ])

    def _build_environment(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        wall_thickness = 0.1
        wall_height = 1.0
        room_size = 5.0

        # Walls
        for bpos, half in [
            ([0,  room_size, wall_height], [room_size, wall_thickness, wall_height]),
            ([0, -room_size, wall_height], [room_size, wall_thickness, wall_height]),
            ([ room_size, 0, wall_height], [wall_thickness, room_size, wall_height]),
            ([-room_size, 0, wall_height], [wall_thickness, room_size, wall_height]),
        ]:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.7, 0.7, 0.7, 1])
            p.createMultiBody(col, vis, basePosition=bpos)

        # Goal position (far enough from spawn)
        while True:
            goal_xy = self._random_position()
            if np.linalg.norm(goal_xy) > 2.0:
                break
        self.goal_position = goal_xy

        # ✅ FIX 2: Obstacles that don't overlap each other or spawn on goal
        self.obstacle_ids = []
        obstacle_positions = []

        for _ in range(5):
            for _ in range(100):  # max attempts
                pos = self._random_position()
                if np.linalg.norm(pos) < 1.0:
                    continue
                if np.linalg.norm(pos - goal_xy) < 1.2:
                    continue
                if any(np.linalg.norm(pos - op) < 1.0 for op in obstacle_positions):
                    continue
                obstacle_positions.append(pos)
                break

            if not obstacle_positions:
                continue

            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.5])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.5], rgbaColor=[1, 0, 0, 1])
            obs_id = p.createMultiBody(
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[obstacle_positions[-1][0], obstacle_positions[-1][1], 0.5]
            )
            self.obstacle_ids.append(obs_id)

        # One obstacle in the middle of path (forces agent to go around)
        mid = (np.array([0.0, 0.0]) + goal_xy) / 2.0
        # Pull it back if it's too close to the goal
        if np.linalg.norm(mid - goal_xy) < 1.5:
            mid = mid * 0.6
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 1])
        mid_id = p.createMultiBody(
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[mid[0], mid[1], 0.5]
        )
        self.obstacle_ids.append(mid_id)

        # Agent
        self.agent_id = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 0.1])

        # Goal visual marker
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.4)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.4, rgbaColor=[0, 1, 0, 1])
        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[goal_xy[0], goal_xy[1], 0.4]
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._build_environment()

        pos, _ = p.getBasePositionAndOrientation(self.agent_id)
        self.prev_distance = np.linalg.norm(
            np.array([pos[0], pos[1]]) - self.goal_position
        )

        return self._get_obs(), {}

    def step(self, action):
        move_speed = 0.1
        turn_speed = 0.15

        pos, orn = p.getBasePositionAndOrientation(self.agent_id)
        yaw = p.getEulerFromQuaternion(orn)[2]

        if action == 0:  # Forward — clamp to room bounds so agent can't escape
            new_x = np.clip(pos[0] + move_speed * np.cos(yaw), -4.7, 4.7)
            new_y = np.clip(pos[1] + move_speed * np.sin(yaw), -4.7, 4.7)
            p.resetBasePositionAndOrientation(self.agent_id, [new_x, new_y, pos[2]], orn)

        elif action == 1:  # Turn left
            yaw += turn_speed
            p.resetBasePositionAndOrientation(
                self.agent_id, pos, p.getQuaternionFromEuler([0, 0, yaw])
            )

        elif action == 2:  # Turn right
            yaw -= turn_speed
            p.resetBasePositionAndOrientation(
                self.agent_id, pos, p.getQuaternionFromEuler([0, 0, yaw])
            )
        # action == 3: STOP — do nothing

        # ✅ FIX 3: Step the simulation so PyBullet actually computes collisions
        p.stepSimulation()

        self.step_count += 1
        obs = self._get_obs()

        pos, _ = p.getBasePositionAndOrientation(self.agent_id)
        agent_xy = np.array([pos[0], pos[1]])
        distance = np.linalg.norm(agent_xy - self.goal_position)

        # ✅ FIX 4: Better reward — distance progress + heading alignment
        delta_dist = self.prev_distance - distance
        reward = delta_dist * 15.0  # reward for getting closer

        # Encourage facing the goal (heading alignment bonus)
        goal_dir = self.goal_position - agent_xy
        goal_angle = np.arctan2(goal_dir[1], goal_dir[0])
        rel_angle = abs(((goal_angle - yaw + np.pi) % (2 * np.pi)) - np.pi)
        heading_bonus = (1.0 - rel_angle / np.pi) * 0.3   # 0.0 to 0.3
        reward += heading_bonus

        # Small step penalty to discourage spinning in place
        reward -= 0.05

        # Collision with obstacle → big penalty + episode ends
        for obs_id in self.obstacle_ids:
            if len(p.getContactPoints(self.agent_id, obs_id)) > 0:
                return obs, -20.0, True, False, {}

        # Reached goal → big reward + episode ends
        if distance < 0.6:
            return obs, 100.0, True, False, {}

        truncated = self.step_count >= self.max_steps
        self.prev_distance = distance

        return obs, reward, False, truncated, {}

    def _get_obs(self):
        image = self._get_camera_image()

        pos, orn = p.getBasePositionAndOrientation(self.agent_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        agent_xy = np.array([pos[0], pos[1]])

        goal_dir = self.goal_position - agent_xy
        distance = np.linalg.norm(goal_dir)
        goal_angle = np.arctan2(goal_dir[1], goal_dir[0])
        rel_angle = (goal_angle - yaw + np.pi) % (2 * np.pi) - np.pi  # -π to π

        goal_info = np.array([
            distance / 8.0,                      # normalized distance (room diagonal ~10)
            np.cos(rel_angle),                   # cosine of bearing to goal
            np.sin(rel_angle),                   # sine of bearing to goal
            self.step_count / self.max_steps,    # episode progress (0→1)
        ], dtype=np.float32)

        return {"image": image, "goal_info": goal_info}

    def _get_camera_image(self):
        width, height = 64, 64

        pos, orn = p.getBasePositionAndOrientation(self.agent_id)
        yaw = p.getEulerFromQuaternion(orn)[2]

        cam_pos = [pos[0], pos[1], pos[2] + 0.6]
        cam_target = [
            pos[0] + np.cos(yaw),
            pos[1] + np.sin(yaw),
            pos[2] + 0.6
        ]

        view_matrix = p.computeViewMatrix(cam_pos, cam_target, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(90, 1.0, 0.01, 10)

        _, _, rgb, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix)
        rgb = np.reshape(rgb, (height, width, 4))[:, :, :3]
        return rgb.astype(np.uint8)

    def close(self):
        p.disconnect(self.client)


register(
    id="IndoorNavRGB-v0",
    entry_point="envs.indoor_nav_env:IndoorNavEnv",
)
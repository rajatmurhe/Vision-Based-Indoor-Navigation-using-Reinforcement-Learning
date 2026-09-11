"""
Vision Based Indoor Navigation using Reinforcement Learning
Presentation and demo dashboard built on top of the existing project.

This file is a standalone Streamlit frontend. It does not modify, import
internals from, or depend on any changes to the existing project files
(envs/, training/, evaluation/, utils/, models/, main.py). It only uses
the existing Gymnasium environment (IndoorNavRGB-v0) and the existing
trained Stable Baselines3 PPO model exactly as they are.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import glob
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import streamlit as st
from PIL import Image

# Heavy, project specific dependencies (gymnasium, stable_baselines3,
# pybullet, plotly) are imported lazily inside the functions that need
# them. That way a missing package produces one friendly message in the
# dashboard instead of crashing the whole process at import time.


# ============================================================================
# CONFIGURATION
#
# This app is not allowed to read or modify the real project files, so a
# handful of internal PyBullet handles (the physics client id, the robot
# body id, the goal position) cannot be imported directly. Instead this
# section lists several common attribute names for each one; the first
# match found on the environment is used. If the real environment uses a
# different name, only the relevant list below needs to change.
# ============================================================================

ENV_ID = "IndoorNavRGB-v0"
MODEL_PATH = os.path.join("models", "ppo_cnn_indoor_nav.zip")
SCREENSHOTS_DIR = "screenshots"
RESULTS_IMAGE_CANDIDATES = [
    "agent_run_results.png",
    os.path.join("screenshots", "agent_run_results.png"),
]

ACTION_NAMES = {0: "FORWARD", 1: "TURN LEFT", 2: "TURN RIGHT", 3: "STOP"}
ACTION_GLYPHS = {0: "\u2191", 1: "\u21b6", 2: "\u21b7", 3: "\u25a0"}
ACTION_COLORS = {
    0: "var(--accent-cyan)",
    1: "var(--accent-amber)",
    2: "var(--accent-amber)",
    3: "var(--accent-red)",
}

DEFAULT_MAX_STEPS = 300
AUTOPLAY_DELAY_SECONDS = 0.12
AGENT_DISPLAY_SIZE = 512

# Candidate attribute names used to locate internal PyBullet handles on the
# unwrapped environment. These only affect the two extra camera views and
# the trajectory plot, never the agent observation, the reward, or the PPO
# model, all of which come straight from the environment and the model
# exactly as written.
CLIENT_ID_ATTRS = [
    "client", "client_id", "_client", "physics_client", "physicsClient",
    "physicsClientId", "cid", "pb_client", "sim_id",
]
ROBOT_ID_ATTRS = [
    "robot_id", "robotId", "agent_id", "agentId", "r2d2_id", "robot",
    "agent_body_id",
]
GOAL_ID_ATTRS = [
    "goal_id", "goalId", "target_id", "goal_body_id", "goal_sphere_id",
    "goal_visual_id",
]
GOAL_POS_ATTRS = [
    "goal_pos", "goal_position", "goal_xyz", "target_pos",
    "target_position", "_goal_pos", "goal_location",
]
OBSTACLE_IDS_ATTRS = ["obstacle_ids", "obstacles", "obstacle_body_ids", "box_ids"]
OBSTACLE_POS_ATTRS = ["obstacle_positions", "obstacle_locations", "box_positions"]
MAX_STEPS_ATTRS = [
    "max_steps", "_max_episode_steps", "max_episode_steps", "horizon",
    "episode_length",
]

# Fallback half size (meters) of the room, used only to frame the top down
# camera when the real arena size cannot be discovered on the environment.
ROOM_HALF_EXTENT_M = 5.0


@dataclass
class EnvInternals:
    client_id: Optional[int] = None
    robot_id: Optional[int] = None
    goal_id: Optional[int] = None
    goal_pos: Optional[tuple] = None
    obstacle_positions: list = field(default_factory=list)
    max_steps: int = DEFAULT_MAX_STEPS
    missing: list = field(default_factory=list)


# ============================================================================
# STYLING
# ============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-deep: #0a0e16;
    --bg-panel: rgba(22, 29, 45, 0.62);
    --border-soft: rgba(140, 170, 255, 0.14);
    --accent-cyan: #4fd1ff;
    --accent-amber: #ffb648;
    --accent-green: #3ee08c;
    --accent-red: #ff5d72;
    --text-primary: #e9edf7;
    --text-muted: #8993ab;
}

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: radial-gradient(circle at 12% 0%, #101a2c 0%, var(--bg-deep) 45%, #060a11 100%);
    color: var(--text-primary);
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.block-container { padding-top: 1.4rem; max-width: 1400px; }

.dash_header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    padding: 1.1rem 1.6rem;
    border-radius: 16px;
    background: var(--bg-panel);
    border: 1px solid var(--border-soft);
    backdrop-filter: blur(14px);
    margin-bottom: 1.4rem;
}
.dash_title { font-size: 1.55rem; font-weight: 700; letter-spacing: 0.03em; margin: 0; }
.dash_subtitle { color: var(--text-muted); font-size: 0.92rem; margin-top: 0.15rem; }

.status_row { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }
.status_pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 0.32rem 0.7rem;
    border-radius: 999px;
    border: 1px solid var(--border-soft);
    background: rgba(255,255,255,0.03);
    color: var(--text-muted);
    letter-spacing: 0.04em;
}
.status_pill.ok { color: var(--accent-green); border-color: rgba(62,224,140,0.35); }
.status_pill.bad { color: var(--accent-red); border-color: rgba(255,93,114,0.35); }

.live_badge {
    display: flex; align-items: center; gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    color: var(--accent-red);
    padding: 0.32rem 0.75rem;
    border-radius: 999px;
    border: 1px solid rgba(255,93,114,0.35);
    background: rgba(255,93,114,0.08);
}
.live_dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent-red);
    animation: pulse 1.6s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255,93,114,0.55); }
    70% { box-shadow: 0 0 0 9px rgba(255,93,114,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,93,114,0); }
}

.panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 1rem 1.1rem 1.2rem 1.1rem;
    backdrop-filter: blur(14px);
    margin-bottom: 1.2rem;
}
.panel_label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent-cyan);
    letter-spacing: 0.09em;
    margin-bottom: 0.15rem;
}
.panel_title { font-size: 1.02rem; font-weight: 600; margin-bottom: 0.6rem; }

.overlay_chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    background: rgba(0,0,0,0.35);
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 8px;
    margin-top: 0.5rem;
    margin-right: 0.4rem;
    border: 1px solid var(--border-soft);
}

.metric_card {
    background: var(--bg-panel);
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: 0.8rem 1rem;
    text-align: left;
    margin-bottom: 0.9rem;
}
.metric_label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.metric_value { font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 600; color: var(--text-primary); }
.metric_value.accent { color: var(--accent-cyan); }
.metric_value.warn { color: var(--accent-amber); }
.metric_value.good { color: var(--accent-green); }
.metric_value.bad { color: var(--accent-red); }

.action_card {
    display: flex; align-items: center; gap: 1.1rem;
    background: var(--bg-panel);
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 1.1rem 1.4rem;
    height: 100%;
}
.action_glyph {
    font-size: 2.2rem;
    line-height: 1;
    width: 3.2rem; height: 3.2rem;
    display: flex; align-items: center; justify-content: center;
    border-radius: 50%;
    border: 1px solid var(--border-soft);
}
.action_name { font-size: 1.15rem; font-weight: 700; letter-spacing: 0.03em; }
.action_sub { color: var(--text-muted); font-size: 0.76rem; margin-top: 0.15rem; font-family: 'JetBrains Mono', monospace; }

.result_banner { border-radius: 18px; padding: 1.2rem 1.5rem; border: 1px solid var(--border-soft); margin-bottom: 1rem; }
.result_banner.good { background: rgba(62,224,140,0.08); border-color: rgba(62,224,140,0.4); }
.result_banner.bad { background: rgba(255,93,114,0.08); border-color: rgba(255,93,114,0.4); }
.result_banner.warn { background: rgba(255,182,72,0.08); border-color: rgba(255,182,72,0.4); }
.result_title { font-size: 1.25rem; font-weight: 700; }

.sysinfo_row { display: flex; justify-content: space-between; padding: 0.35rem 0; border-bottom: 1px solid var(--border-soft); font-size: 0.86rem; }
.sysinfo_row:last-child { border-bottom: none; }
.sysinfo_key { color: var(--text-muted); }
.sysinfo_val { font-family: 'JetBrains Mono', monospace; color: var(--text-primary); }

section[data-testid="stSidebar"] { background: #0b111c; border-right: 1px solid var(--border-soft); }

.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid var(--border-soft);
    background: rgba(79,209,255,0.08);
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
    padding: 0.55rem 0.6rem;
}
.stButton > button:hover { border-color: var(--accent-cyan); color: var(--accent-cyan); }
.stProgress > div > div > div { background-color: var(--accent-cyan); }
[data-testid="stImage"] img { border-radius: 12px; }
</style>
"""


# ============================================================================
# ENVIRONMENT / MODEL LOADING (cached so they are created exactly once)
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    from stable_baselines3 import PPO
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model file was not found at {model_path}")
    return PPO.load(model_path)


@st.cache_resource(show_spinner=False)
def load_environment(env_id: str):
    import gymnasium as gym
    try:
        # Importing the project's env module registers IndoorNavRGB-v0 as
        # a side effect in most Gymnasium project layouts. This import is
        # only a safety net, the id may already be registered elsewhere
        # (for example by main.py), in which case this simply does nothing.
        import envs.indoor_nav_env  # noqa: F401
    except Exception:
        pass

    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except TypeError:
        env = gym.make(env_id)

    # The environment is expected to run its own PyBullet client, headless
    # (DIRECT mode), exactly as it already does during training and
    # evaluation. This app never calls p.connect itself and never creates
    # a second simulation, it only reads from the client the environment
    # already owns (see resolve_env_internals below).
    return env


def _unwrap(env) -> Any:
    """Return the innermost environment instance beneath any Gymnasium wrappers."""
    return getattr(env, "unwrapped", env)


def _first_attr(obj: Any, names: list) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def resolve_env_internals(env) -> EnvInternals:
    """
    Look up the PyBullet client id, the robot body id, and the goal
    position on the real environment so the extra camera views and the
    trajectory plot can read the same simulation state the agent is
    seeing. Nothing here is invented, every value either comes straight
    off the environment object or from a live PyBullet query using ids
    found on that object.
    """
    raw = _unwrap(env)
    internals = EnvInternals()

    internals.client_id = _first_attr(raw, CLIENT_ID_ATTRS)
    internals.robot_id = _first_attr(raw, ROBOT_ID_ATTRS)
    goal_id = _first_attr(raw, GOAL_ID_ATTRS)
    internals.goal_id = goal_id

    goal_pos = _first_attr(raw, GOAL_POS_ATTRS)
    if goal_pos is None and goal_id is not None and internals.client_id is not None:
        try:
            import pybullet as p
            pos, _ = p.getBasePositionAndOrientation(goal_id, physicsClientId=internals.client_id)
            goal_pos = pos
        except Exception:
            goal_pos = None
    internals.goal_pos = tuple(goal_pos) if goal_pos is not None else None

    obstacle_positions = _first_attr(raw, OBSTACLE_POS_ATTRS)
    if obstacle_positions is None:
        obstacle_ids = _first_attr(raw, OBSTACLE_IDS_ATTRS)
        if obstacle_ids is not None and internals.client_id is not None:
            try:
                import pybullet as p
                collected = []
                for oid in obstacle_ids:
                    pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=internals.client_id)
                    collected.append(pos)
                obstacle_positions = collected
            except Exception:
                obstacle_positions = []
    internals.obstacle_positions = list(obstacle_positions) if obstacle_positions else []

    spec_max = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    max_steps = spec_max if spec_max else _first_attr(raw, MAX_STEPS_ATTRS)
    internals.max_steps = int(max_steps) if max_steps else DEFAULT_MAX_STEPS

    missing = []
    if internals.client_id is None:
        missing.append("PyBullet client id")
    if internals.robot_id is None:
        missing.append("robot body id")
    if internals.goal_pos is None:
        missing.append("goal position")
    internals.missing = missing
    return internals


def get_robot_pose(internals: Optional[EnvInternals]):
    if internals is None or internals.client_id is None or internals.robot_id is None:
        return None, None
    try:
        import pybullet as p
        pos, orn = p.getBasePositionAndOrientation(internals.robot_id, physicsClientId=internals.client_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return pos, yaw
    except Exception:
        return None, None


def _render_pybullet_camera(client_id, eye, target, up, width, height, fov=60.0):
    import pybullet as p
    view = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=up)
    proj = p.computeProjectionMatrixFOV(fov=fov, aspect=width / height, nearVal=0.05, farVal=30.0)
    _, _, rgb, _, _ = p.getCameraImage(
        width, height, view, proj,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id,
    )
    arr = np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)
    return arr


def get_third_person_view(internals: Optional[EnvInternals], width=480, height=360) -> Optional[np.ndarray]:
    """A chase camera placed behind and above the robot, built from the
    same PyBullet client and the same body position the agent itself is
    using this step, so it always matches the agent POV frame."""
    if internals is None or internals.client_id is None or internals.robot_id is None:
        return None
    pos, yaw = get_robot_pose(internals)
    if pos is None or yaw is None:
        return None
    back, up_offset = 1.8, 1.3
    eye = [pos[0] - back * np.cos(yaw), pos[1] - back * np.sin(yaw), pos[2] + up_offset]
    target = [pos[0], pos[1], pos[2] + 0.2]
    try:
        return _render_pybullet_camera(internals.client_id, eye, target, [0, 0, 1], width, height, fov=65.0)
    except Exception:
        return None


def get_top_down_view(internals: Optional[EnvInternals], width=420, height=420) -> Optional[np.ndarray]:
    """An overhead camera looking straight down at the room, centered on
    the robot's current position, built from the same PyBullet client."""
    if internals is None or internals.client_id is None:
        return None
    pos, _ = get_robot_pose(internals)
    cx, cy = (pos[0], pos[1]) if pos else (0.0, 0.0)
    cam_height = max(ROOM_HALF_EXTENT_M * 2.2, 6.0)
    eye = [cx, cy, cam_height]
    target = [cx, cy, 0.0]
    try:
        return _render_pybullet_camera(internals.client_id, eye, target, [0, 1, 0], width, height, fov=70.0)
    except Exception:
        return None


def classify_outcome(info: dict, terminated: bool, truncated: bool, obs) -> str:
    """
    Turn the step result into one of the five statuses the dashboard
    shows. Custom Gymnasium environments name their info dict keys
    differently, so several common names are checked first; if none of
    them are present, a fallback based on the same normalized distance
    the policy was trained on is used, which is real observation data
    rather than an invented value.
    """
    info = info or {}
    for key in ("is_success", "success", "goal_reached", "reached_goal"):
        if info.get(key):
            return "GOAL REACHED"
    for key in ("collision", "is_collision", "collided", "crashed"):
        if info.get(key):
            return "COLLISION"
    if truncated and not terminated:
        return "TIMEOUT"
    if terminated:
        try:
            dist = float(np.asarray(obs["goal_info"])[0])
            if dist < 0.08:
                return "GOAL REACHED"
        except Exception:
            pass
        return "COLLISION"
    return "STOPPED"


def status_css_class(status: str) -> str:
    return {
        "GOAL REACHED": "good",
        "COLLISION": "bad",
        "TIMEOUT": "warn",
        "STOPPED": "warn",
        "NAVIGATING": "accent",
    }.get(status, "")


# ============================================================================
# SESSION STATE
# ============================================================================

def init_state():
    defaults = dict(
        obs=None,
        internals=None,
        running=False,
        deterministic=True,
        step_delay=AUTOPLAY_DELAY_SECONDS,
        step_count=0,
        total_reward=0.0,
        current_reward=0.0,
        trajectory=[],
        current_action=None,
        status="STANDBY",
        episode_num=0,
        done=True,
        last_result=None,
        boot_error=None,
    )
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_episode(new_episode: bool):
    env = load_environment(ENV_ID)
    obs, _info = env.reset()
    internals = resolve_env_internals(env)
    pos, _ = get_robot_pose(internals)

    st.session_state.obs = obs
    st.session_state.internals = internals
    st.session_state.step_count = 0
    st.session_state.total_reward = 0.0
    st.session_state.current_reward = 0.0
    st.session_state.current_action = None
    st.session_state.trajectory = [tuple(pos[:2])] if pos else []
    st.session_state.status = "NAVIGATING"
    st.session_state.done = False
    st.session_state.running = False
    st.session_state.last_result = None
    if new_episode:
        st.session_state.episode_num += 1


def do_step():
    if st.session_state.done or st.session_state.obs is None:
        return
    env = load_environment(ENV_ID)
    model = load_model(MODEL_PATH)

    action, _ = model.predict(st.session_state.obs, deterministic=st.session_state.deterministic)
    action = int(np.asarray(action).reshape(-1)[0])
    obs, reward, terminated, truncated, info = env.step(action)

    st.session_state.obs = obs
    st.session_state.current_action = action
    st.session_state.step_count += 1
    st.session_state.current_reward = float(reward)
    st.session_state.total_reward += float(reward)

    internals = st.session_state.internals
    pos, _ = get_robot_pose(internals)
    if pos:
        st.session_state.trajectory.append(tuple(pos[:2]))

    if terminated or truncated:
        st.session_state.done = True
        st.session_state.running = False
        status = classify_outcome(info, terminated, truncated, obs)
        st.session_state.status = status
        try:
            final_distance = float(np.asarray(obs["goal_info"])[0])
        except Exception:
            final_distance = None
        st.session_state.last_result = dict(
            status=status,
            steps=st.session_state.step_count,
            total_reward=st.session_state.total_reward,
            final_distance=final_distance,
        )
    else:
        st.session_state.status = "NAVIGATING"


# ============================================================================
# RENDERING
# ============================================================================

def render_header(model_ok: bool, env_ok: bool):
    model_cls, model_txt = ("ok", "LOADED") if model_ok else ("bad", "UNAVAILABLE")
    env_cls, env_txt = ("ok", "ACTIVE") if env_ok else ("bad", "OFFLINE")
    st.markdown(
        f"""
        <div class="dash_header">
            <div>
                <p class="dash_title">VISION BASED INDOOR NAVIGATION</p>
                <p class="dash_subtitle">Autonomous navigation using deep reinforcement learning</p>
            </div>
            <div class="status_row">
                <span class="status_pill {model_cls}">PPO MODEL: {model_txt}</span>
                <span class="status_pill {env_cls}">PYBULLET: {env_txt}</span>
                <span class="status_pill">ENVIRONMENT: {ENV_ID}</span>
                <span class="live_badge"><span class="live_dot"></span>LIVE SIMULATION</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model_ok: bool, env_ok: bool):
    with st.sidebar:
        st.markdown('<div class="panel_label">CONTROL PANEL</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel_title">SIMULATION CONTROLS</div>', unsafe_allow_html=True)

        system_disabled = not (model_ok and env_ok)
        has_episode = st.session_state.obs is not None

        if st.button("START NEW EPISODE", disabled=system_disabled):
            try:
                with st.spinner("Resetting environment..."):
                    reset_episode(new_episode=True)
            except Exception as exc:
                st.session_state.boot_error = str(exc)

        run_label = "PAUSE" if st.session_state.running else "RUN"
        run_disabled = system_disabled or not has_episode or st.session_state.done
        if st.button(run_label, disabled=run_disabled):
            st.session_state.running = not st.session_state.running

        step_disabled = system_disabled or not has_episode or st.session_state.done or st.session_state.running
        if st.button("STEP ONCE", disabled=step_disabled):
            try:
                do_step()
            except Exception as exc:
                st.session_state.boot_error = str(exc)

        if st.button("RESET", disabled=system_disabled or not has_episode):
            try:
                reset_episode(new_episode=False)
            except Exception as exc:
                st.session_state.boot_error = str(exc)

        if st.button("STOP", disabled=system_disabled or not has_episode):
            st.session_state.running = False
            st.session_state.done = True
            st.session_state.status = "STOPPED"

        st.markdown("<hr style='border-color: rgba(140,170,255,0.14);'>", unsafe_allow_html=True)

        st.session_state.deterministic = st.checkbox(
            "DETERMINISTIC POLICY",
            value=st.session_state.deterministic,
            help="When enabled the PPO policy always takes its highest probability action instead of sampling.",
        )
        st.session_state.step_delay = st.slider(
            "STEP DELAY (SECONDS)",
            min_value=0.0, max_value=0.4,
            value=st.session_state.step_delay, step=0.02,
            help="Delay between automatic steps while RUN is active.",
        )

        st.markdown("<hr style='border-color: rgba(140,170,255,0.14);'>", unsafe_allow_html=True)
        st.caption(f"Episode {st.session_state.episode_num}")

        if st.session_state.boot_error:
            st.error("A system error was reported, see details below.")
            with st.expander("Technical details"):
                st.code(st.session_state.boot_error)


def render_camera_panels(internals: Optional[EnvInternals]):
    left, right = st.columns([1.35, 1])
    obs = st.session_state.obs
    agent_img = None
    if isinstance(obs, dict) and "image" in obs:
        agent_img = np.asarray(obs["image"]).astype(np.uint8)

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel_label">AGENT POV</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel_title">FIRST PERSON VISUAL OBSERVATION</div>', unsafe_allow_html=True)
        if agent_img is not None:
            upscaled = Image.fromarray(agent_img).resize(
                (AGENT_DISPLAY_SIZE, AGENT_DISPLAY_SIZE), Image.NEAREST
            )
            st.image(upscaled, use_container_width=True)
        else:
            st.info("The agent camera feed appears once an episode starts.")
        st.markdown(
            f'<span class="overlay_chip">CAMERA 64 x 64</span>'
            f'<span class="overlay_chip">AGENT VIEW</span>'
            f'<span class="overlay_chip">STEP {st.session_state.step_count}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel_label">THIRD PERSON VIEW</div>', unsafe_allow_html=True)
        tp_img = get_third_person_view(internals)
        if tp_img is not None:
            st.image(tp_img, use_container_width=True)
        else:
            st.caption("This camera turns on once an episode starts.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel_label">TOP DOWN MAP</div>', unsafe_allow_html=True)
        td_img = get_top_down_view(internals)
        if td_img is not None:
            st.image(td_img, use_container_width=True)
        else:
            st.caption("This camera turns on once an episode starts.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_telemetry(internals: Optional[EnvInternals]):
    obs = st.session_state.obs
    goal_info = None
    if isinstance(obs, dict) and "goal_info" in obs:
        goal_info = np.asarray(obs["goal_info"]).astype(float)

    distance_m = None
    if internals and internals.goal_pos:
        pos, _ = get_robot_pose(internals)
        if pos:
            distance_m = float(np.hypot(pos[0] - internals.goal_pos[0], pos[1] - internals.goal_pos[1]))

    if distance_m is not None:
        distance_label = f"{distance_m:.2f} m"
    elif goal_info is not None:
        distance_label = f"{goal_info[0]:.2f} (normalized)"
    else:
        distance_label = "N/A"

    direction_label = "N/A"
    if goal_info is not None and len(goal_info) >= 3:
        angle_deg = float(np.degrees(np.arctan2(goal_info[2], goal_info[1])))
        if -30 <= angle_deg <= 30:
            compass = "AHEAD"
        elif 30 < angle_deg <= 150:
            compass = "LEFT"
        elif -150 <= angle_deg < -30:
            compass = "RIGHT"
        else:
            compass = "BEHIND"
        direction_label = f"{compass} ({angle_deg:.0f} deg)"

    action = st.session_state.current_action
    action_label = ACTION_NAMES.get(action, "IDLE") if action is not None else "IDLE"
    max_steps = internals.max_steps if internals else DEFAULT_MAX_STEPS

    metrics = [
        ("STEP", f"{st.session_state.step_count} / {max_steps}", ""),
        ("MAX STEPS", f"{max_steps}", ""),
        ("CURRENT REWARD", f"{st.session_state.current_reward:+.3f}", "accent"),
        ("TOTAL REWARD", f"{st.session_state.total_reward:+.3f}", "accent"),
        ("DISTANCE TO GOAL", distance_label, ""),
        ("GOAL DIRECTION", direction_label, ""),
        ("CURRENT ACTION", action_label, "warn"),
        ("EPISODE STATUS", st.session_state.status, status_css_class(st.session_state.status)),
    ]

    cols = st.columns(4)
    for i, (label, value, tone) in enumerate(metrics):
        with cols[i % 4]:
            value_cls = f"metric_value {tone}".strip()
            st.markdown(
                f'<div class="metric_card"><div class="metric_label">{label}</div>'
                f'<div class="{value_cls}">{value}</div></div>',
                unsafe_allow_html=True,
            )


def render_action_indicator():
    action = st.session_state.current_action
    glyph = ACTION_GLYPHS.get(action, "\u2022")
    name = ACTION_NAMES.get(action, "AWAITING FIRST STEP") if action is not None else "AWAITING FIRST STEP"
    color = ACTION_COLORS.get(action, "var(--text-muted)")
    mode_txt = "DETERMINISTIC" if st.session_state.deterministic else "STOCHASTIC"
    st.markdown(
        f"""
        <div class="action_card">
            <div class="action_glyph" style="color:{color}; border-color:{color};">{glyph}</div>
            <div>
                <div class="action_name" style="color:{color};">{name}</div>
                <div class="action_sub">POLICY OUTPUT, PPO {mode_txt} INFERENCE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_progress(internals: Optional[EnvInternals]):
    max_steps = internals.max_steps if internals else DEFAULT_MAX_STEPS
    frac = min(st.session_state.step_count / max_steps, 1.0) if max_steps else 0.0
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel_label">EPISODE PROGRESS</div>', unsafe_allow_html=True)
    st.progress(frac)
    st.caption(f"{st.session_state.step_count} / {max_steps} steps completed, status {st.session_state.status}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_trajectory_plot(internals: Optional[EnvInternals]):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel_label">TRAJECTORY</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel_title">AGENT PATH IN THE ARENA</div>', unsafe_allow_html=True)

    traj = st.session_state.trajectory
    if not traj:
        st.info("The trajectory plot fills in once an episode starts stepping.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly is not installed, showing raw coordinates instead.")
        st.write(traj)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    xs = [pt[0] for pt in traj]
    ys = [pt[1] for pt in traj]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers", name="Path",
        line=dict(color="#4fd1ff", width=2), marker=dict(size=4, color="#4fd1ff"),
    ))
    fig.add_trace(go.Scatter(
        x=[xs[0]], y=[ys[0]], mode="markers", name="Start",
        marker=dict(size=12, color="#3ee08c", symbol="diamond"),
    ))
    fig.add_trace(go.Scatter(
        x=[xs[-1]], y=[ys[-1]], mode="markers", name="Agent",
        marker=dict(size=13, color="#ffb648", symbol="circle"),
    ))
    if internals and internals.goal_pos:
        fig.add_trace(go.Scatter(
            x=[internals.goal_pos[0]], y=[internals.goal_pos[1]], mode="markers", name="Goal",
            marker=dict(size=15, color="#3ee08c", symbol="star"),
        ))
    if internals and internals.obstacle_positions:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in internals.obstacle_positions],
            y=[p[1] for p in internals.obstacle_positions],
            mode="markers", name="Obstacles",
            marker=dict(size=11, color="#ff5d72", symbol="square"),
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e9edf7", family="JetBrains Mono"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False, title="X (m)"),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False, title="Y (m)",
            scaleanchor="x", scaleratio=1,
        ),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)
    if internals and internals.goal_pos is None:
        st.caption("Goal marker not shown, this app could not find a goal position attribute on the environment.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_episode_result():
    result = st.session_state.last_result
    if not result:
        return
    status = result["status"]
    tone = {"GOAL REACHED": "good", "COLLISION": "bad", "TIMEOUT": "warn", "STOPPED": "warn"}.get(status, "warn")
    dist_txt = f'{result["final_distance"]:.3f}' if result.get("final_distance") is not None else "N/A"
    outcome_txt = "SUCCESS" if status == "GOAL REACHED" else "FAILURE"
    outcome_cls = "good" if outcome_txt == "SUCCESS" else "bad"

    st.markdown(
        f'<div class="result_banner {tone}"><div class="result_title">{status}</div></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric_card"><div class="metric_label">TOTAL STEPS</div><div class="metric_value">{result["steps"]}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric_card"><div class="metric_label">TOTAL REWARD</div><div class="metric_value accent">{result["total_reward"]:.3f}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric_card"><div class="metric_label">FINAL DISTANCE</div><div class="metric_value">{dist_txt}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric_card"><div class="metric_label">OUTCOME</div><div class="metric_value {outcome_cls}">{outcome_txt}</div></div>', unsafe_allow_html=True)


def render_system_info():
    with st.expander("SYSTEM INFORMATION", expanded=False):
        rows = [
            ("Algorithm", "PPO, Proximal Policy Optimization"),
            ("Observation", "RGB visual input plus goal information"),
            ("Visual input", "64 x 64 RGB"),
            ("Actions", "4 discrete navigation actions"),
            ("Environment", "PyBullet and Gymnasium"),
            ("Policy", "Stable Baselines3 PPO"),
        ]
        html = "".join(
            f'<div class="sysinfo_row"><span class="sysinfo_key">{k}</span><span class="sysinfo_val">{v}</span></div>'
            for k, v in rows
        )
        st.markdown(f'<div class="panel">{html}</div>', unsafe_allow_html=True)


def render_results_gallery():
    with st.expander("EXPERIMENT RESULTS", expanded=False):
        images = []
        for path in RESULTS_IMAGE_CANDIDATES:
            if os.path.exists(path) and path not in images:
                images.append(path)
        if os.path.isdir(SCREENSHOTS_DIR):
            shots = sorted(
                glob.glob(os.path.join(SCREENSHOTS_DIR, "*.png"))
                + glob.glob(os.path.join(SCREENSHOTS_DIR, "*.jpg"))
            )
            images.extend([s for s in shots if s not in images])

        if not images:
            st.caption("No screenshots or results image found yet in the project folder.")
            return

        cols = st.columns(3)
        for i, path in enumerate(images):
            with cols[i % 3]:
                try:
                    st.image(path, use_container_width=True, caption=os.path.basename(path))
                except Exception:
                    st.caption(f"Could not load {os.path.basename(path)}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.set_page_config(
        page_title="Vision Based Indoor Navigation",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_state()

    model_ok, env_ok = True, True
    try:
        load_model(MODEL_PATH)
    except Exception as exc:
        model_ok = False
        st.session_state.boot_error = str(exc)
    try:
        load_environment(ENV_ID)
    except Exception as exc:
        env_ok = False
        st.session_state.boot_error = str(exc)

    render_header(model_ok, env_ok)
    render_sidebar(model_ok, env_ok)

    if not (model_ok and env_ok):
        st.error(
            "The navigation system could not start. Check that the trained model file and "
            "the custom Gymnasium environment are available next to this app, then reload the page."
        )
        if st.session_state.boot_error:
            with st.expander("Technical details"):
                st.code(st.session_state.boot_error)
        render_system_info()
        return

    internals = st.session_state.internals
    render_camera_panels(internals)
    render_telemetry(internals)

    col_a, col_b = st.columns([1, 1.4])
    with col_a:
        render_action_indicator()
    with col_b:
        render_progress(internals)

    render_trajectory_plot(internals)
    render_episode_result()
    render_system_info()
    render_results_gallery()

    if st.session_state.running and not st.session_state.done:
        try:
            do_step()
        except Exception as exc:
            st.session_state.running = False
            st.session_state.boot_error = str(exc)
        else:
            time.sleep(st.session_state.step_delay)
            st.rerun()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # last resort safety net, never show a raw traceback by default
        st.error("Something went wrong while rendering the dashboard.")
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
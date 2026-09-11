from gymnasium.envs.registration import register

register(
    id="IndoorNavRGB-v0",
    entry_point="envs.indoor_nav_env:IndoorNavEnv",
)

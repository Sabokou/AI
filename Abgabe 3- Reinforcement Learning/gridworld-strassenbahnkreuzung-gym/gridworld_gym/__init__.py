from gym.envs.registration import register

register(
    id="gridworld-v0",
    entry_point="gridworld_gym.envs:GridWorldEnv",
    max_episode_steps=399
)
from gymnasium.envs.registration import register

register(
     id="pacmangym/PacManEnv-v0",
     entry_point="pacmangym.envs:PacManEnv",
     max_episode_steps=1000,
)

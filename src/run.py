import sys
import time
import numpy as np
import pacmangym
import gymnasium as gym

def main():
    np.set_printoptions(threshold=sys.maxsize)
    single_env('human')
    #multi_envs('human')


def single_env(render_mode):
    env = gym.make('pacmangym/PacManEnv-v0', render_mode=render_mode)
    env = gym.wrappers.FlattenObservation(env)
    observation, info = env.reset()
    for _ in range(1000000):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()

def multi_envs(render_mode):
    envs = gym.vector.make('pacmangym/PacManEnv-v0', num_envs=20, render_mode=render_mode)
    #env_fns = [lambda: gym.make("pacmangym/PacManEnv-v0", render_mode=render_mode)] * 10
    #envs = gym.vector.AsyncVectorEnv(env_fns, shared_memory=True)
    observation, info = envs.reset()

    for _ in range(1000000):
        observation, reward, terminated, truncated, info = envs.step(envs.action_space.sample())

    if terminated or truncated:
        observation, info = envs.reset()

    envs.close()

if __name__ == "__main__":
    main()

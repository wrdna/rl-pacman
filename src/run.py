import pacmangym
import gymnasium as gym

def main():
    env = gym.make('pacmangym/PacManEnv-v0', render_mode="human")
    observation, info = env.reset()
    for i in range(1000):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()

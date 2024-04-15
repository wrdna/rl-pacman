import numpy as np
import gymnasium as gym
import random
import time
import math

def main():
    env = gym.make("Taxi-v3").env
    env.reset()
    env.render()

    print("Action Space {}".format(env.action_space))
    print("State Space {}".format(env.observation_space))

    alpha = 0.7
    discount_factor = 0.618
    epsilon = 1
    min_epsiolon = 0.01
    decay = 0.01

    train_episodes = 2000
    test_episodes - 100
    max_steps = 100

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    training_rewards = []
    epsilons = []

    for episode in range(train_episodes):
        # Resetting env each loop
        state = env.reset()

        # Tracking rewards
        total_training_rewards = 0

        for step in range(100):
            # Choose an action give the states based on a random number
            exp_exp_tradeoff = random.uniform(0, 1)

            # If the random number is larger than epsilon: emplying exploitation
            # and selecting best action
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(Q[state, :])
            
            # Otherwise, employing exploration: choosing a random action
            else:
                action = env.action_space.sample()

            # Perform action and get reward
            new_state, reward, done, info = env.step(action)
            
            # Update Q-table with Bellman Equation
            Q[state, action] = Q[state, action] + alpha * (reward + discount_factor *
                                                           np.max(Q[new_state, :]) - Q[state, action])





if __name__ == "__main__":
    main()

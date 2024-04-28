# Written By RK
# This agent implements Q learning to train a model to play pacman
import torch
import random
import numpy as np
from collections import deque
from agentmodel import Linear_QNet, QTrainer
from helper import plot
from game import GameController
from constants import LEFT, RIGHT, UP, DOWN, STOP, SCATTER, CHASE, FREIGHT, SPAWN, DIRECTION_MAP


# Define Rewards
Rewards = {
    "MOVE": -1, 
    "STOPPED": -50,
    "EAT_PELLET": 50,
    "EAT_POWER_PELLET": 100,
    "EAT_GHOST": 100,
    "BEAT_LEVEL": 1000,
    "DIE": -1000,
    "EAT_FRUIT": 50
}

Settings = {
    "FPS": 60,
    "T_MULT": 2,
    "DISABLE_GHOSTS": True,
    "DISABLE_LEVELS": True,
}

# agent params
MAX_MEMORY = 1_000_000
BATCH_SIZE = 5000
LR = 0.001

# adds randomness into training early and decays over time to 0
RAND_SIZE = 200

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9   # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(13, 1024, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        # debugging
        self.resetdirections()

    # must be encoded to booleans
    def get_state(self, game):
        
        pacman_u = game.pacman.direction == UP
        pacman_d = game.pacman.direction == DOWN
        pacman_r = game.pacman.direction == RIGHT
        pacman_l = game.pacman.direction == LEFT
        pacman_s = game.pacman.direction == STOP

        # calculate target pellet direction
        target_pellet = game.getClosestPellet()

        state = [
            # pacman direction
            pacman_u,
            pacman_d,
            pacman_r,
            pacman_l,
            pacman_s,

            game.pacman.canGoUp(),
            game.pacman.canGoDown(),
            game.pacman.canGoRight(),
            game.pacman.canGoLeft(),

            # closest pellet
            target_pellet.position.x > game.pacman.position.x, # pellet right
            target_pellet.position.x < game.pacman.position.x, # pellet left
            target_pellet.position.y > game.pacman.position.y, # pellet up
            target_pellet.position.y < game.pacman.position.y, # pellet down

            # closest power pellet

        ]
         # ghosts
        # for ghost in game.ghosts:
            # print(ghost)

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if > MAX_MEM

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves tradeoff exploring/exploitation
        self.epsilon = RAND_SIZE - self.n_games
        move = [0, 0, 0, 0] # encode as array
        if random.randint(0, 200) < self.epsilon:
            r = random.randint(0, 3)
            move[r] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            m = torch.argmax(prediction).item()
            move[m] = 1
            if m == 0 :
                self.up += 1
            elif m == 1:
                self.down += 1
            elif m == 2:
                self.left += 1
            elif m == 3:
                self.right += 1
            # print(f"Model Predicted {DIRECTION_MAP[decodeDirection(move)]}")
        # convert to pacman game inputs
        return move

    def random_action(self):
        return random.randint(-2, 2)
    
    def resetdirections(self):
        self.up = 0
        self.down = 0
        self.right = 0
        self.left = 0

    def printDirectionCount(self):
        print(f"U {self.up} | D {self.down} | R {self.right} | L {self.left} ")


def train():
    plot_scores = []
    plot_mean_scores = []
    plot_avg_10 = []
    total_score = 0
    record = 0
    # agent
    agent = Agent()
    accum_reward = 0

    # game
    game = GameController(Rewards, Settings)
    game.startGame()
    # run 
    while True:
        # get current state
        c_state = agent.get_state(game)

        # calculate move
        move = agent.get_action(c_state)

        # do move
        reward, gameover, score = game.update(decodeDirection(move))
        new_state = agent.get_state(game)
        accum_reward += reward

        # train short memory
        agent.train_short_memory(c_state, move, reward, new_state, gameover)

        # remember - don't forget
        agent.remember(c_state, move, reward, new_state, gameover)

        if gameover:
            # reset
            game.restartGame()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            # log
            print(f"game {agent.n_games} | score {score} | reward {accum_reward}")
            agent.printDirectionCount()
            agent.resetdirections()

            accum_reward = 0
            
            total_score += score
            mean_score = total_score / agent.n_games

            plot_scores.append(score)
            plot_mean_scores.append(mean_score)
            plot_avg_10.append(sum(plot_scores[-10:]) / 10  if len(plot_scores) >= 10 else 0)

            plot(plot_scores, plot_mean_scores, plot_avg_10)



def play_model():
    pass

# encode direction as array
def decodeDirection(matrix):
    # up
    if np.array_equal([1, 0, 0, 0], matrix):
        return 1
    # down
    elif np.array_equal([0, 1, 0, 0], matrix):
        return -1
    # left
    elif np.array_equal([0, 0, 1, 0], matrix):
        return 2
    # right
    elif np.array_equal([0, 0, 0, 1], matrix):
        return -2



if __name__ == '__main__':
    train()






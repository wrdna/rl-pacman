# Written By RK
# This agent implements Q learning to train a model to play pacman
import random
from run import GameController

# Define Rewards
Rewards = {
    "MOVE": -1, 
    "EAT_PELLET": 2,
    "EAT_POWER_PELLET": 5,
    "EAT_GHOST": 20,
    "BEAT_LEVEL": 100,
    "DIE": -200,
    "GAMEOVER": -200,
    "EAT_FRUIT": 10
}


# Agent Params
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
TRAIN_RAND_SIZE = 80

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9   # discount rate
        # self.memory = deque(maxlen=MAX_MEMORY)
        # self.model = Linear_QNet(11, 256, 3)
        # self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        return None

    def remember(self):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self):
        pass

    def get_action(self, state):
        move = random.randint(-2, 2)
        return move


def train():
    agent = Agent()
    games_played = 0
    accum_reward = 0

    # game
    pacman = GameController()
    pacman.startGame()
    # run 
    while True:

        # get state
        state = agent.get_state(pacman)

        # calculate move
        move = agent.get_action(state)

        reward, gameover, score = pacman.update(move)
        accum_reward += reward

        if gameover:
            games_played += 1

            pacman.restartGame()

            print(f"game {games_played} | score {score} | reward {accum_reward}")
            accum_reward = 0

def play_model(self):
    pass


if __name__ == '__main__':
    train()






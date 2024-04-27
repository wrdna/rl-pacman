# Written By RK
# This agent implements Q learning to train a model to play pacman
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

    def get_state(self):
        pass

    def remember(self):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self):
        pass

    def action(self):
        pass


def train(self):
    pacman = GameController()

def play_model(self):
    pass


if __name__ == '__main__':
    train()






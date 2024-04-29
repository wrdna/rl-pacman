# PyTorch Tutorial for DQN
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gymnasium as gym
import math
import time
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch. optim as optim
import torch.nn.functional as F

import pacmangym

MEMORY_BUFFER_SIZE = 300000
NUM_EPISODES = 30000
BATCH_SIZE = 1024 
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 3000000
TAU = 0.005
LR = 1e-4
SHOWCASE_MODEL=True
MODEL='models/policy_net_11300.pt'

env = gym.make("pacmangym/PacManEnv-v0", render_mode='human')
env = gym.wrappers.FlattenObservation(env)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

memory = ReplayMemory(MEMORY_BUFFER_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

steps_done = 0

n_actions = env.action_space.n
state, info = env.reset()

n_observations = len(state)
policy_net = DQN(n_observations, n_actions).to(device)
if SHOWCASE_MODEL:
    policy_net = torch.load(MODEL, map_location=torch.device('cpu')) 
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

episode_score = []
episode_reward = []

def main():
    if torch.cuda.is_available():
        num_episodes = NUM_EPISODES 
    else:
        num_episodes = 50
    total_t = 0 
    progbar = tqdm(range(num_episodes))
    for i_episode in progbar:
        if i_episode%100 == 0:
            torch.save(policy_net,f'models/policy_net_{i_episode}.pt') 
        env = gym.make("pacmangym/PacManEnv-v0", render_mode='human')
        env = gym.wrappers.FlattenObservation(env)
        #else:
        #    env = gym.make("pacmangym/PacManEnv-v0")
        #    env = gym.wrappers.FlattenObservation(env)
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            progbar.set_description(f'total steps: {total_t}')
            total_t+=1
            action = select_action(state)
            observation, reward, terminated, truncated, info = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
    
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
    
            # Move to the next state
            state = next_state
    
            # Perform one step of the optimization (on the policy network)
            optimize_model()
    
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
    
            if done:
                episode_score.append(info['score'])
                episode_reward.append(info['total_reward'])
                plot_score()
                plot_reward()
                break
    print('Complete')
    plot_score(show_result=True)
    plot_reward(show_result=True)
    plt.ioff()
    plt.show()

def select_action(state):
    if SHOWCASE_MODEL:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def plot_reward(show_result=False):
    plt.figure(2)
    reward_t = torch.tensor(episode_reward, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_t.numpy())
    # Take 100 episode averages and plot them too
    if len(reward_t) >= 100:
        means = reward_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # Take 1000 episode averages and plot them too
    if len(reward_t) >= 1000:
        means = reward_t.unfold(0, 1000, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(999), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def plot_score(show_result=False):
    plt.figure(1)
    score_t = torch.tensor(episode_score, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(score_t.numpy())
    # Take 100 episode averages and plot them too
    if len(score_t) >= 100:
        means = score_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # Take 1000 episode averages and plot them too
    if len(score_t) >= 1000:
        means = score_t.unfold(0, 1000, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(999), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if __name__ == "__main__":
    main()

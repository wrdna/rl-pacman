# This agent implements Q deep learning to train a model to play pacman
# Written by RK, based on snake game model implemented by Patrick Loeber on freeCodeCamp.org YouTube
# Source Video https://www.youtube.com/watch?v=L8ypSXwyBds&t=3s&ab_channel=freeCodeCamp.org
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # activation function
        x = F.relu(self.linear1(x)) 
        # x = F.leaky_relu(self.linear1(x))
        x = self.linear2(x) # applies to net
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        file_name = os.path.join("./model", file_name)
        torch.load(file_name)


class QTrainer:
    
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.L1Loss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1 predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
            

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward(loss)

        self.optimizer.step()

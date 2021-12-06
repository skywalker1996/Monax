import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple

class DQN(nn.Module):

    def __init__(self, seq_len, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(seq_len, 50)
        self.fc2 = nn.Linear(50, 30)
        self.head = nn.Linear(30, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # print('x.size() = ', x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.head(x)

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        """保存一次交互
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
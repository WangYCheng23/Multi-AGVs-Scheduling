import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class Agent(nn.Module):
    def __init__(self, hidden_size, num_observation, num_actions):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(num_observation, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(num_observation, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,1)
        )

    # def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
    #     torch.nn.init.orthogonal_(layer.weight, std)
    #     torch.nn.init.constant_(layer.bias, bias_const)
    #     return layer

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        # hidden = self.network(x / 255.0)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from qap.actor import CombinedActorNet
from qap.critic import CriticNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent(nn.Module):
    def __init__(self, input_dim_technical, input_dim_behavior, output_dim_technical, output_dim_behavior, action_dim, hidden_dim, gamma=0.99, tau=1e-3):
        super(DDPGAgent, self).__init__()
        self.actor = CombinedActorNet(input_dim_technical, input_dim_behavior, output_dim_technical, output_dim_behavior, action_dim).to(device)
        self.actor_target = CombinedActorNet(input_dim_technical, input_dim_behavior, output_dim_technical, output_dim_behavior, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        state_dim = input_dim_technical + input_dim_behavior
        self.critic = CriticNet(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = CriticNet(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.tau = tau

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state).squeeze().cpu().numpy()
        return np.clip(action, -1.0, 1.0)

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        state_batch = torch.FloatTensor([e[0] for e in batch]).to(device)
        action_batch = torch.FloatTensor([e[1] for e in batch]).to(device)
        reward_batch = torch.FloatTensor([e[2] for e in batch]).to(device)
        next_state_batch = torch.FloatTensor([e[3] for e in batch]).to(device)
        done_batch = torch.FloatTensor([e[4] for e in batch]).to(device)

        # Update critic network
        next_action_batch = self.actor_target(next_state_batch).detach()
        q_next = self.critic_target(next_state_batch, next_action_batch).squeeze()
        q_target = reward_batch + self.gamma * (1 - done_batch) * q_next
        q_current = self.critic(state_batch, action_batch).squeeze()
        critic_loss = F.mse_loss(q_current, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor network
        action_pred = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, action_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


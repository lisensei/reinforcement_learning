import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gym
from collections import namedtuple, deque
import scipy.stats as st
import random

transition = namedtuple("transition", ["state", "action", "reward", "next_state", "done"])


class QNET(nn.Module):
    def __init__(self, state_size):
        super(QNET, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        if type(x) is not torch.Tensor:
            x = torch.Tensor(x)
        output = self.layers(x)
        return output


num_actions = 2
policy_net = QNET(4)
behavior_net = QNET(4)
steps = 10000
env = gym.envs.make("CartPole-v1")
state = env.reset()
loss = nn.MSELoss()
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
tds = []
plt.ion()
plt.show()
position_max_abs = 4.8
cv_max_abs = 0.835
angle_max_abs = 0.418
av_max_abs = 1.273
for e in range(steps):
    state[0] = state[0] / position_max_abs
    state[1] = state[1] / cv_max_abs
    state[2] = state[2] / angle_max_abs
    state[3] = state[3] / av_max_abs
    behavior_state_actions = behavior_net(state)
    eps = 1 / (e + 10)
    probilities = np.ones(num_actions) * eps / num_actions
    action = torch.argmax(behavior_state_actions).numpy()
    probilities[action] = 1 - eps + eps / num_actions
    action_rv = st.rv_discrete(values=(np.arange(2), probilities))
    action_selected = action_rv.rvs()
    q_state_actions = policy_net(state)
    q_state_action = q_state_actions[action_selected]
    state, reward, done, _ = env.step(action)
    with torch.no_grad():
        q_next_state_action = torch.argmax(policy_net(state))
    if done:
        td_target = torch.tensor(reward)
        state = env.reset()
    else:
        td_target = q_next_state_action + reward
    temporal_difference = loss(td_target, q_state_action)
    optimizer.zero_grad()
    temporal_difference.backward()
    optimizer.step()
    x = np.arange(0, e + 1)
    tds.append(temporal_difference.detach().numpy())
    if e % 100 == 0:
        test_env = gym.envs.make("CartPole-v1")
        state = test_env.reset()
        end = False
        test_reward = 0
        while not end:
            with torch.no_grad():
                qvalues = policy_net(state)
            action = torch.argmax(qvalues).numpy()
            state, reward, end, _ = test_env.step(action)
            test_reward += reward
            test_env.render()
        print(f"test reward:{test_reward}")
        behavior_net.load_state_dict(policy_net.state_dict())

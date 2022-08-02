"""
Deep Q learning on Cartpole-v1
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gym
from collections import namedtuple, deque
import scipy.stats as st
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument("-steps", default=20000)
parser.add_argument("-bnet_update_rate", default=10)
parser.add_argument("-composite_state_length", default=5)
parser.add_argument("-hidden_size", default=256)
parser.add_argument("-eps", default=0.3)
parser.add_argument("-gamma", default=1)
parser.add_argument("-learning_rate", default=1e-2)
parser.add_argument("-show_rate", default=50)
parser.add_argument("-batch_size", default=16)
parameters = parser.parse_args()

transition = namedtuple("transition", ["state", "action", "reward", "next_state", "done"])

'''Linear deep Q net'''


class QNET(nn.Module):
    def __init__(self, state_size):
        super(QNET, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 2),

        )

    def forward(self, x):
        return self.layers(x)


class Experience:
    def __init__(self, maxlen=10000):
        self.experience = deque(maxlen=maxlen)

    def __len__(self):
        return len(self.experience)

    def __getitem__(self, index):
        return self.experience[index]

    def sample(self, batch_size=16):
        indices = random.sample(range(0, len(self.experience)), batch_size)
        batch = [self.experience[i] for i in indices]
        return batch

    def append(self, x):
        self.experience.append(x)

    def sort(self, key=None):
        self.experience = sorted(self.experience, key)

    def replay(self, qnet, target_net, loss_fn, optimizer):
        transitions = self.sample()
        states = torch.tensor(np.array([t.state for t in transitions]))
        actions = torch.tensor(np.array([t.action for t in transitions]), dtype=torch.long)
        next_states = torch.tensor(np.array([t.next_state for t in transitions]))
        rewards = torch.tensor(np.array([t.reward for t in transitions]), dtype=torch.float32)
        terminal_states = np.array([t.done for t in transitions])
        ts = np.argwhere(terminal_states == True).reshape(1, -1)
        '''Computes Q(s,a)'''
        action_values = target_net(states)
        action_values = action_values[np.arange(len(action_values)), actions]
        '''Computes max of Q(s',a')'''
        with torch.no_grad():
            next_action_values = qnet(next_states)
            next_action_values, _ = torch.max(next_action_values, 1)
        td_target = rewards + next_action_values
        td_target[ts] = rewards[ts]
        tds = loss_fn(td_target, action_values)
        optimizer.zero_grad()
        tds.backward()
        optimizer.step()
        return tds.item()


@torch.no_grad()
def test(net):
    test_env = gym.envs.make("CartPole-v1")
    test_state = test_env.reset()
    end = False
    test_reward = 0
    while not end:
        qvalues = net(torch.tensor(test_state))
        action = torch.argmax(qvalues).numpy()
        test_state, r, end, _ = test_env.step(action)
        test_reward += r
        test_env.render()
    test_env.close()
    return test_reward


def normalize_state(state):
    position_max_abs = 4.8
    cv_max_abs = 0.835
    angle_max_abs = 0.418
    av_max_abs = 1.273
    state[0] = state[0] / position_max_abs
    state[1] = state[1] / cv_max_abs
    state[2] = state[2] / angle_max_abs
    state[3] = state[3] / av_max_abs
    return state


num_actions = 2
state_size = 4
'''
policy net is the actual net that learns parameters.
behavior net generates episodes.
'''
q_net = QNET(state_size, )
q_target_net = QNET(state_size)
experience = Experience()

steps = parameters.steps
env = gym.envs.make("CartPole-v1")
state = env.reset()
loss = nn.MSELoss()
optimizer = torch.optim.Adam(q_target_net.parameters(), lr=parameters.learning_rate)
tds = []
Returns = []
test_returns = []
plt.ion()
fig, axes = plt.subplots(3, 1, constrained_layout=True)
plt.show()

'''n is the frequency at which the behavior net copies policy net's parameters'''
Return = 0
episode = 0
num_test = 0
'''max sequence length for recurrent dqn'''
state_deque = deque(maxlen=parameters.composite_state_length)
for e in range(parameters.steps):
    '''computes state action values while following behavior net'''
    exp_state = np.copy(state)
    evolving_state_actions = q_target_net(torch.tensor(state))
    eps = parameters.eps ** ((e + 200) / 200)
    probabilities = np.ones(num_actions) * eps / num_actions

    '''computes the actions that evolving net should take'''
    max_action_index = torch.argmax(evolving_state_actions).numpy()

    probabilities[max_action_index] = 1 - eps + eps / num_actions
    action_rv = st.rv_discrete(values=(np.arange(2), probabilities))
    action_selected = action_rv.rvs()

    '''evolving net interacts with the environment'''
    state, reward, done, _ = env.step(action_selected)
    exp_transition = transition(exp_state, action_selected, reward, np.copy(state), done)
    Return += reward
    experience.append(exp_transition)
    if done:
        state = env.reset()
        Returns.append(Return)
        Return = 0
        episode += 1
        axes[0].plot(np.arange(episode), Returns)
        axes[0].set_xlabel("episode")
        axes[0].set_ylabel("return")
        axes[0].set_title("Episode Return")
    if e > parameters.batch_size:
        td = experience.replay(q_net, q_target_net, loss, optimizer)
        tds.append(td)
        axes[1].plot(np.arange(len(tds)), tds)
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Replay Loss")
    plt.pause(0.00001)
    if e % parameters.bnet_update_rate == 0:
        q_net.load_state_dict(q_target_net.state_dict())
    if e % parameters.show_rate == 0:
        tr = test(q_target_net)
        test_returns.append(tr)
        num_test += 1
        s = np.arange(num_test)
        axes[2].plot(s, np.array(test_returns))
        axes[2].set_xlabel("episode")
        axes[2].set_ylabel("return")
        axes[2].set_title("Test Episode-Return")

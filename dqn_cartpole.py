"""
Deep Q learning on Cartpole-v1
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gym
from collections import namedtuple, deque
import scipy.stats as st
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-steps", default=20000)
parser.add_argument("-bnet_update_rate", default=10)
parser.add_argument("-composite_state_length", default=5)
parser.add_argument("-hidden_size", default=128)
parser.add_argument("-eps", default=0.5)
parser.add_argument("-gamma", default=1)
parser.add_argument("-learning_rate", default=1e-2)
parser.add_argument("-show_rate", default=100)
parameters = parser.parse_args()

transition = namedtuple("transition", ["state", "action", "reward", "next_state", "done"])

'''Recurrent deep Q net, with plain RNN and LSTM available'''


class RQNET(nn.Module):
    def __init__(self, state_size, hidden_size, output_size):
        super(RQNET, self).__init__()
        self.output_size = output_size
        self.input_to_hidden = nn.Linear(state_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size + state_size, hidden_size)
        self.activator = torch.rand(hidden_size).reshape(hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.lstm = nn.LSTM(input_size=state_size, num_layers=1, hidden_size=hidden_size, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=hidden_size, num_layers=1, hidden_size=hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x, lstm=True):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        if len(x.shape) < 1:
            x = x.unsqueeze(0)
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if not lstm:
            hidden_input = torch.cat([x[0], self.activator])
            hidden_state = torch.relu(self.hidden_to_hidden(hidden_input))
            x = x[1:, :]
            for s in x:
                hidden_input = torch.cat([s, hidden_state])
                hidden_state = torch.relu(self.hidden_to_hidden(hidden_input))
            output = torch.relu(self.output(hidden_state))
        else:
            x = x.unsqueeze(0)
            hidden_state, _ = self.lstm(x)
            hidden_state, _ = self.lstm1(hidden_state)
            output = self.output(hidden_state[-1, -1])
            output = output.reshape(num_actions)
        return output


'''Linear deep Q net'''


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


def test(policy_net):
    test_env = gym.envs.make("CartPole-v1")
    test_state = test_env.reset()
    if not is_rnn:
        test_state = test_state[-1]
    end = False
    test_reward = 0
    while not end:
        test_state_deque.append(torch.tensor(test_state))
        test_deque_len = len(test_state_deque)
        test_state = torch.cat(list(test_state_deque)).reshape(test_deque_len, -1)
        with torch.no_grad():
            qvalues = policy_net(test_state)
        action = torch.argmax(qvalues).numpy()
        test_state, reward, end, _ = test_env.step(action)
        test_reward += reward
        test_env.render()
    print(f"test reward:{test_reward}")
    test_env.close()


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
policy_net = RQNET(state_size, parameters.hidden_size, num_actions)
behavior_net = RQNET(state_size, parameters.hidden_size, num_actions)

is_rnn = isinstance(policy_net, RQNET)
steps = parameters.steps
env = gym.envs.make("CartPole-v1")
state = env.reset()
loss = nn.MSELoss()
optimizer = torch.optim.Adam(policy_net.parameters(), lr=parameters.learning_rate)
tds = []
Returns = []
plt.ion()
fig, axes = plt.subplots(2, 1)
plt.show()

'''n is the frequency at which the behavior net copies policy net's parameters'''
Return = 0
episode = 0

'''max sequence length for recurrent dqn'''
state_deque = deque(maxlen=parameters.composite_state_length)
test_state_deque = deque(maxlen=parameters.composite_state_length)
for e in range(parameters.steps):
    state_deque.append(torch.tensor(state))
    deque_len = len(state_deque)
    state = torch.cat(list(state_deque)).reshape(deque_len, -1)
    if not is_rnn:
        state = state[-1]
    with torch.no_grad():
        '''computes state action values while following behavior net'''
        behavior_state_actions = behavior_net(state)
    eps = parameters.eps ** (1000 / (e + 1000))
    probilities = np.ones(num_actions) * eps / num_actions

    '''computes the actions that policy net should take'''
    max_action_index = torch.argmax(behavior_state_actions).numpy()

    probilities[max_action_index] = 1 - eps + eps / num_actions
    action_rv = st.rv_discrete(values=(np.arange(2), probilities))
    action_selected = action_rv.rvs()

    '''computes Q(s,a)'''
    q_state_actions = policy_net(state)
    q_state_action = q_state_actions[action_selected]

    '''behavior net interacts with the environment'''
    state, reward, done, _ = env.step(action_selected)

    Return += reward
    with torch.no_grad():
        ''' computes max of Q(s',a')'''
        q_next_state_action = torch.argmax(policy_net(torch.tensor(state)))
    if done:
        td_target = torch.tensor(reward)
        state = env.reset()
        state_deque.clear()
        Returns.append(Return)
        Return = 0
        episode += 1
        es = np.arange(episode)
        axes[1].plot(es, Returns)
    else:
        td_target = q_next_state_action + reward

    '''Computes temporal difference'''
    temporal_difference = loss(td_target, q_state_action)
    optimizer.zero_grad()
    temporal_difference.backward()
    optimizer.step()

    '''Plot staff'''
    x = np.arange(0, e + 1)
    tds.append(temporal_difference.detach().numpy())
    axes[0].plot(x, tds)
    plt.pause(0.00001)
    if e % parameters.bnet_update_rate == 0:
        behavior_net.load_state_dict(policy_net.state_dict())
    if e % parameters.show_rate == 0:
        test(policy_net)

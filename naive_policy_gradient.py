"""This script runs the naive policy gradient algorithm on CartPole-v1"""
import torch
import torch.nn as nn
import numpy as np
import scipy.stats as st
import gym
from collections import deque, namedtuple
import torch.utils.data as utils
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-saving_threshold", default=400, type=int)
parser.add_argument("-env_name", default="CartPole-v1")
parser.add_argument("-epochs", default=10000)
parser.add_argument("-fit_count", default=10)
script_parameters = parser.parse_args()

Transition = namedtuple("transition", ["state", "action", "reward", "next_state"])

'''Normalizing state. However, it seems not necessary for Cartpole-v1'''


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


'''
Trajectory in a batch have different lengths. 
This function turns it into tuples of tensors (states,action,reward)
'''


def collate_batch(array):
    states = torch.cat([ele[0] for ele in array])
    actions = torch.cat([ele[1] for ele in array])
    reward = torch.cat([ele[2] for ele in array])
    return states, actions, reward


class Net(nn.Module):
    def __init__(self, state_size, action_size):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_size),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class TrajectoryDataset:
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        episode = self.trajectories[index]
        states = torch.tensor(np.array([t.state for t in episode]))
        actions = [torch.tensor(t.action).reshape(-1) for t in episode]
        actions = torch.cat(actions).reshape(-1)
        total_reward = (len(episode) - 1) * torch.ones(actions.shape)
        return states, actions, total_reward


class Agent:
    def __init__(self, env: gym.envs, mem_size=160):
        self.env = env
        self.brain = Net(env.observation_space.shape[0], env.action_space.n)
        self.mem_size = mem_size
        self.memory = deque(maxlen=mem_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.brain.parameters())

    def generate_data(self, ):
        self.memory.clear()
        sample_env = gym.make(script_parameters.env_name)
        num_actions = sample_env.action_space.n
        for i in range(self.mem_size):
            episode = []
            state = sample_env.reset()
            done = False
            while not done:
                state_copy = np.copy(state)
                output = self.brain(torch.tensor(state))
                action_sampler = st.rv_discrete(
                    values=(np.arange(num_actions), torch.softmax(output, 0).detach().numpy()))
                action = action_sampler.rvs()
                state, reward, done, _ = sample_env.step(action)
                transition = Transition(state_copy, action, reward, np.copy(state))
                episode.append(transition)
            self.memory.append(episode)
        sample_env.close()

    def policy_gradient_learning(self, epochs, fit_count=10, saving_threshold=400):
        plt.ion()
        fig, axe = plt.subplots()
        returns = []
        batch_size = 4
        for e in range(epochs):
            self.generate_data()
            dataset = TrajectoryDataset(self.memory)
            dataloader = utils.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)
            '''Fits the data sampled from the old policy'''
            for o in range(fit_count):
                for i, (state, action, total_return) in enumerate(dataloader):
                    output = self.brain(state)
                    ''' 
                    Grad at time step t of a specific trajectory k  is:
                    grad_t = R_k* grad(p(a|s))
                    where R_k is the total return of trajectory k.
                    The gradient of is the sum of grad at all time step in a batch.
                    '''
                    loss = torch.sum(total_return * self.loss_fn(output, action))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            if e % 1 == 0:
                reward = self.test()
                returns.append(reward)
                axe.plot(np.arange(len(returns)), np.array(returns))
                plt.pause(0.0001)
                if reward > 200:
                    Rs = []
                    for _ in range(10):
                        Rs.append(self.test())
                    mean_reward = np.mean(np.array(Rs))
                    if mean_reward > saving_threshold:
                        torch.save(self.brain.state_dict(), "CartPole-v1.pth")

    '''test policy's performance'''

    @torch.no_grad()
    def test(self):
        test_env = gym.make(script_parameters.env_name)
        state = test_env.reset()
        done = False
        reward = 0
        while not done:
            output = self.brain(torch.tensor(state))
            action = torch.argmax(output)
            state, _, done, _ = test_env.step(action.numpy())
            reward += 1
            test_env.render()
        test_env.close()
        return reward


env = gym.envs.make(script_parameters.env_name)
agent = Agent(env)
agent.generate_data()
agent.policy_gradient_learning(script_parameters.epochs, script_parameters.fit_count,
                               script_parameters.saving_threshold)

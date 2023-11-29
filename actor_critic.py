"""This script runs the actor-critic policy gradient algorithm on LunarLander-v2"""
import torch
import torch.nn as nn
import numpy as np
import scipy.stats as st
import gym
from collections import deque, namedtuple
import torch.utils.data as utils
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-saving_threshold", default=400, type=int)
parser.add_argument("-env_name", default="LunarLander-v2")
parser.add_argument("-epochs", default=10000)
parser.add_argument("-fit_count", default=10)
parser.add_argument("-gamma", default=1, type=float)
parser.add_argument("-memory_size", default=160,type=int)
script_parameters = parser.parse_args()

Transition = namedtuple("transition", ["state", "action", "reward", "next_state"])

'''
Trajectory in a batch have different lengths. 
This function turns it into tuples of tensors (states,action,reward)
'''


class TrajectoryDataset:
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        episode = self.trajectories[index]
        states = torch.tensor(np.array([t.state for t in episode]))
        actions = torch.tensor([t.action for t in episode]).reshape(-1)
        rewards = torch.tensor([t.reward for t in episode]).reshape(-1, 1)
        next_states = torch.tensor(np.array([t.next_state for t in episode]))
        return states, actions, rewards, next_states


def collate_batch(array):
    states = torch.cat([element[0] for element in array])
    actions = torch.cat([element[1] for element in array])
    rewards = torch.cat([element[2] for element in array])
    next_states = torch.cat([element[3] for element in array])
    '''
    total_returns: a list of G_t at time step t
    G_t=R_t+gamma^{1}*R_{t+1}+gamma^2*R_{t+2}+...+gamma^{T-t}*R_T
    '''
    total_returns = []
    episode_length = len(rewards)
    for step in range(episode_length):
        gammas = torch.pow(script_parameters.gamma, torch.arange(0, episode_length - step))
        remaining_rewards = rewards[step:]
        cumsum = torch.sum(remaining_rewards * gammas.reshape(remaining_rewards.shape))
        total_returns.append(cumsum)
    total_returns = torch.tensor(total_returns).reshape(-1, 1)
    return states, actions, rewards, next_states, total_returns


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


class ValueNet(nn.Module):
    def __init__(self, state_size):
        super(ValueNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class Agent:
    def __init__(self, env: gym.envs, mem_size=script_parameters.memory_size):
        self.env = env
        self.brain = Net(env.observation_space.shape[0], env.action_space.n)
        self.value_estimator = ValueNet(env.observation_space.shape[0])
        self.mem_size = mem_size
        self.memory = deque(maxlen=mem_size)
        self.action_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.value_loss_fn = nn.SmoothL1Loss()
        self.action_optimizer = torch.optim.Adam(self.brain.parameters())
        self.value_optimizer = torch.optim.Adam(self.value_estimator.parameters())

    def generate_data(self, ):
        self.memory.clear()
        sample_env = gym.make(script_parameters.env_name)
        num_actions = sample_env.action_space.n
        for i in range(self.mem_size):
            episode = []
            state,_ = sample_env.reset()
            done = False
            while not done:
                state_copy = np.copy(state)
                output = self.brain(torch.tensor(state))
                action_sampler = st.rv_discrete(
                    values=(np.arange(num_actions), torch.softmax(output, 0).detach().numpy()))
                action = action_sampler.rvs()
                state, reward, done, _,_ = sample_env.step(action)
                transition = Transition(state_copy, action, reward, np.copy(state))
                episode.append(transition)
            self.memory.append(episode)
        sample_env.close()

    @torch.no_grad()
    def get_value_estimates(self, states):
        self.value_estimator.eval()
        return self.value_estimator(states)

    def policy_gradient_learning(self, epochs, fit_count=script_parameters.fit_count, saving_threshold=400):
        plt.ion()
        fig, axe = plt.subplots(2, 1, layout="constrained")
        returns = []
        batch_size = 4
        epoch_loss = []
        for e in tqdm(range(epochs)):
            self.generate_data()
            dataset = TrajectoryDataset(self.memory)
            actor_dataloader = utils.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)
            critic_dataloader = utils.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)
            self.value_estimator.train()
            for _ in range(fit_count):
                for c, (current_state, _, _, _, expected_return) in enumerate(critic_dataloader):
                    estimates = self.value_estimator(current_state)
                    value_estimator_loss = self.value_loss_fn(estimates, expected_return)
                    self.value_optimizer.zero_grad()
                    value_estimator_loss.backward()
                    self.value_optimizer.step()
            '''Fits the data sampled from the old policy'''
            self.value_estimator.eval()
            self.brain.train()
            for o in range(fit_count):
                fit_loss = []
                for i, (state, action, reward, next_state, total_return) in enumerate(actor_dataloader):
                    output = self.brain(state)
                    ''' 
                    Grad at time step t of a specific trajectory k  is:
                    grad_t = R_k* grad(p(a|s))
                    where R_k is the total return of trajectory k.
                    The gradient of is the sum of grad at all time step in a batch.
                    '''
                    reward_estimate = total_return
                    # reward_estimate = self.get_value_estimates(next_state) + reward
                    baseline = self.get_value_estimates(state)
                    advantage = reward_estimate - baseline
                    # loss = torch.sum(advantage.squeeze(1) * self.action_loss_fn(output, action))
                    loss = torch.mean(advantage.squeeze(1) * self.action_loss_fn(output, action))
                    self.action_optimizer.zero_grad()
                    loss.backward()
                    self.action_optimizer.step()
                    fit_loss.append(loss.detach().numpy())
            epoch_loss.append(np.mean(fit_loss))
            test_reward = self.test()
            returns.append(test_reward)
            axe[0].set_title("test run return")
            axe[1].set_title("averaged epoch loss")
            axe[0].plot(np.arange(len(returns)), np.array(returns))
            axe[1].plot(np.arange(len(epoch_loss)), epoch_loss)
            plt.pause(0.0001)
            mean_reward = np.mean(np.array(returns))
            if mean_reward > saving_threshold:
                torch.save(self.brain.state_dict(), "CartPole-v1.pth")

    '''test policy's performance'''

    @torch.no_grad()
    def test(self):
        self.brain.train()
        test_env = gym.make(script_parameters.env_name)
        state,_ = test_env.reset()
        done = False
        reward = 0
        while not done:
            output = self.brain(torch.tensor(state))
            action = torch.argmax(output)
            state, _, done, _,_ = test_env.step(action.numpy())
            reward += 1
            test_env.render()
        test_env.close()
        return reward


if __name__ == "__main__":
    env = gym.envs.make(script_parameters.env_name)
    agent = Agent(env)
    agent.policy_gradient_learning(script_parameters.epochs, script_parameters.fit_count,
                                   script_parameters.saving_threshold)

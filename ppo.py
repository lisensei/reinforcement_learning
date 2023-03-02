"""This script implements the proximal policy gradient algorithm on LunarLander-v2"""
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
parser.add_argument("-saving_threshold", default=100, type=int)
parser.add_argument("-env_name", default="LunarLander-v2")
parser.add_argument("-epochs", default=100)
parser.add_argument("-critic_fit_count", default=10)
parser.add_argument("-gamma", default=1, type=float)
parser.add_argument("-memory_size", default=100, type=int)
parser.add_argument("-parameter_sync_frequency", default=5, type=int)
parser.add_argument("-actor_learning_rate", default=1e-3)
parser.add_argument("-critic_learning_rate", default=1e-3)
hp = parser.parse_args()

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
        actions = torch.tensor([t.action for t in episode]).reshape(-1).to(torch.int64)
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
        gammas = torch.pow(hp.gamma, torch.arange(0, episode_length - step))
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
            nn.BatchNorm1d(16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
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
    def __init__(self, env: gym.envs, mem_size=hp.memory_size, actor_learning_rate=hp.actor_learning_rate,
                 critic_learning_rate=hp.critic_learning_rate):
        self.env = env
        self.policy_net = Net(env.observation_space.shape[0], env.action_space.n)
        self.behavior_net = Net(env.observation_space.shape[0], env.action_space.n)
        self.behavior_net.load_state_dict(self.policy_net.state_dict())
        for param in self.behavior_net.parameters():
            param.requires_grad = False
        self.value_estimator = ValueNet(env.observation_space.shape[0])
        self.mem_size = mem_size
        self.memory = deque(maxlen=mem_size)
        self.action_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.value_loss_fn = nn.SmoothL1Loss()
        self.action_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=actor_learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_estimator.parameters(), lr=critic_learning_rate)

    @torch.no_grad()
    def generate_data(self, importance_sampling=True):
        if importance_sampling:
            self.behavior_net.eval()
        else:
            self.policy_net.eval()
        self.memory.clear()
        sample_env = gym.make(hp.env_name)
        num_actions = sample_env.action_space.n
        for i in range(self.mem_size):
            episode = []
            state, _ = sample_env.reset()
            done = False
            truncated = False
            while not done and not truncated:
                state_copy = np.copy(state)
                if importance_sampling:
                    output = self.behavior_net(torch.tensor(state).unsqueeze(0)).squeeze(0)
                else:
                    output = self.policy_net(torch.tensor(state).unsqueeze(0)).squeeze(0)
                action_sampler = st.rv_discrete(
                    values=(np.arange(num_actions), torch.softmax(output, 0).detach().numpy()))
                action = action_sampler.rvs()
                state, reward, done, _, _ = sample_env.step(action)
                transition = Transition(state_copy, action, reward, np.copy(state))
                episode.append(transition)
            self.memory.append(episode)
        sample_env.close()

    @torch.no_grad()
    def get_value_estimates(self, states):
        self.value_estimator.eval()
        return self.value_estimator(states)

    def PPO(self, epochs, critic_fit_count=hp.critic_fit_count, parameter_sync_frequency=hp.parameter_sync_frequency,
            saving_threshold=400, eps=0.2):
        plt.ion()
        fig, axe = plt.subplots(3, 1, layout="constrained")
        returns = []
        time_steps = []
        batch_size = 4
        epoch_loss = []
        for e in tqdm(range(epochs)):
            self.generate_data()
            dataset = TrajectoryDataset(self.memory)
            actor_dataloader = utils.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)
            critic_dataloader = utils.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)
            self.value_estimator.train()
            for _ in range(critic_fit_count):
                for c, (current_state, _, _, _, expected_return) in enumerate(critic_dataloader):
                    estimates = self.value_estimator(current_state)
                    value_estimator_loss = self.value_loss_fn(estimates, expected_return)
                    self.value_optimizer.zero_grad()
                    value_estimator_loss.backward()
                    self.value_optimizer.step()
            '''Fits the data sampled from the old policy'''
            self.value_estimator.eval()
            self.policy_net.train()
            for o in range(parameter_sync_frequency):
                fit_loss = []
                for i, (state, action, reward, next_state, total_return) in enumerate(actor_dataloader):
                    self.policy_net.train()
                    output = self.policy_net(state)
                    probabilities = torch.softmax(output, dim=1)
                    with torch.no_grad():
                        # self.behavior_net.load_state_dict(self.policy_net.state_dict())
                        # self.behavior_net.train()
                        # policy_net_action_probability, _ = torch.max(probabilities, dim=1)
                        # predicted_action = torch.argmax(probabilities, keepdim=True, dim=1)
                        self.behavior_net.eval()
                        behavior_net_action_probabilities = torch.softmax(self.behavior_net(state), dim=1).gather(1,
                                                                                                                  action.unsqueeze(
                                                                                                                      1)).squeeze(
                            1)
                        self.behavior_net.eval()
                    policy_net_action_probability = probabilities.gather(1, action.unsqueeze(1)).squeeze(1)
                    importance_ratio = policy_net_action_probability / behavior_net_action_probabilities
                    ''' 
                    Grad at time step t of a specific trajectory k  is:
                    grad_t = R_k* grad(p(a|s))
                    where R_k is the total return of trajectory k.
                    The gradient of is the sum of grad at all time step in a batch.
                    '''
                    reward_estimate = total_return
                    baseline = self.get_value_estimates(state)
                    advantage = (reward_estimate - baseline).squeeze(1)
                    # PPO
                    adjusted_advantage = torch.min(importance_ratio * advantage,
                                                   torch.clip(importance_ratio, 1 - eps,
                                                              1 + eps) * advantage)
                    # loss = torch.mean(
                    #    adjusted_advantage / policy_net_action_probability * self.action_loss_fn(output, action))
                    loss = -torch.mean(adjusted_advantage)
                    self.action_optimizer.zero_grad()
                    loss.backward()
                    self.action_optimizer.step()
                    fit_loss.append(loss.detach().numpy())
            self.behavior_net.load_state_dict(self.policy_net.state_dict())
            epoch_loss.append(np.mean(fit_loss))
            test_reward, steps_taken = self.test()
            returns.append(test_reward)
            time_steps.append(steps_taken)
            axe[0].set_title("test run return")
            axe[1].set_title("time steps taken")
            axe[2].set_title("averaged epoch loss")
            axe[0].plot(np.arange(len(returns)), np.array(returns))
            axe[1].plot(np.arange(len(time_steps)), np.array(time_steps))
            axe[2].plot(np.arange(len(epoch_loss)), epoch_loss)
            plt.pause(1)
            mean_reward = np.mean(np.array(returns))
            print(test_reward)
            if test_reward > saving_threshold:
                torch.save(self.policy_net.state_dict(), f"{hp.env_name}.pth")

    '''test policy's performance'''

    @torch.no_grad()
    def test(self):
        self.policy_net.eval()
        test_env = gym.make(hp.env_name, render_mode="human")
        state, _ = test_env.reset()
        done = False
        truncated = False
        reward = 0
        steps = 0
        while not done and not truncated:
            output = self.policy_net(torch.tensor(state).unsqueeze(0)).squeeze(0)
            action = torch.argmax(output)
            state, r, done, truncated, _ = test_env.step(action.numpy())
            reward += r
            steps += 1
            test_env.render()
        test_env.close()
        return reward, steps


if __name__ == "__main__":
    env = gym.envs.make(hp.env_name)
    agent = Agent(env)
    agent.PPO(hp.epochs, hp.critic_fit_count, hp.parameter_sync_frequency,
              hp.saving_threshold)

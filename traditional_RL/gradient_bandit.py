from argparse import Action
import numpy as np
import scipy.stats as st
import torch
np.set_printoptions(precision=4)


class Bandit():
    def __init__(self, num_arm=10) -> None:
        self.reward_mean = st.norm.rvs(size=num_arm)
        self.reward_sampler = []
        for i in range(num_arm):
            self.reward_sampler.append(st.norm(self.reward_mean[i]))

    def run(self, action):
        return self.reward_sampler[action].rvs()


class Agent():
    def __init__(self, num_arm=10) -> None:
        self.num_arm = num_arm
        self.action_space = np.arange(0, num_arm).reshape(num_arm, 1)
        self.policy = np.array(np.ones((num_arm, 1))/num_arm)
        self.qvalue_estimates = np.zeros((num_arm, 1))
        self.action_selection_count = np.zeros((num_arm, 1))
        self.action_sampler = st.rv_discrete(
            values=(self.action_space, self.policy))
        self.reward_recived = 0
        self.last_action = -1
        self.last_reward = 0

    def play(self, bandit):
        action = self.action_sampler.rvs()
        self.last_action = action
        self.action_selection_count[action] += 1
        return bandit.run(action)

    def greedy_update(self, eps=1e-1):
        self.qvalue_estimates[self.last_action] = self.qvalue_estimates[self.last_action] + \
            (self.last_reward-self.qvalue_estimates[self.last_action]
             )/self.action_selection_count[self.last_action]
        self.policy = np.ones(self.policy.shape)*eps/(self.num_arm-1)
        self.policy[np.argmax(self.qvalue_estimates)] = np.array(1-eps)
        self.action_sampler = st.rv_discrete(
            values=(self.action_space, self.policy))

    def gradient_update(self, lr=1e-3):
        self.qvalue_estimates[self.last_action] = self.qvalue_estimates[self.last_action] + \
            (self.last_reward-self.qvalue_estimates[self.last_action]
             )/self.action_selection_count[self.last_action]
        temp = self.policy[self.last_action] + lr*(self.last_reward -
                                                   self.qvalue_estimates[self.last_action])*(1-self.policy[self.last_action])
        self.policy = self.policy-lr * \
            (self.last_reward-self.qvalue_estimates[self.last_action])
        self.policy[self.last_action] = temp
        self.policy = self.softmax(self.policy)
    def softmax(self, a):
        numerator = np.exp(a)
        sum = np.sum(numerator)
        return numerator/sum

    def reset(self,):
        self.policy = np.array(np.ones((self.num_arm, 1))/self.num_arm)
        self.qvalue_estimates = np.zeros((self.num_arm, 1))
        self.action_selection_count = np.zeros((self.num_arm, 1))
        self.action_sampler = st.rv_discrete(
            values=(self.action_space, self.policy))
        self.last_action = -1
        self.last_reward = 0



band = Bandit()
bandman = Agent()
for i in range(1000):
    reward = bandman.play(band)
    bandman.last_reward = reward
    bandman.greedy_update()
print(f"true mean:{band.reward_mean}\n")
print(f"Epsilon greedy:\nestimates: {bandman.qvalue_estimates.reshape(-1)} \npolicy: {bandman.policy.reshape(-1)}\naction count:{bandman.action_selection_count.reshape(-1)}")

bandman.reset()
print("\n\n")
for i in range(1000):
    reward = bandman.play(band)
    bandman.last_reward = reward
    bandman.gradient_update(lr=4)

print(f"Gradient:\nestimates: {bandman.qvalue_estimates.reshape(-1)} \npolicy: {bandman.policy.reshape(-1)}\naction count: {bandman.action_selection_count.reshape(-1)}")

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)


class KBandits:
    def __init__(self, number_of_bandits, eps):
        self.num_actions = number_of_bandits
        self.reward_mean = st.norm(loc=0, scale=1).rvs(size=self.num_actions).reshape(10)
        self.actions = np.arange(0, self.num_actions).astype(np.float64).reshape(10)
        self.action_counts = np.zeros(self.num_actions, dtype=np.int32)
        self.reward_estimates = np.zeros(self.num_actions)
        self.eps = np.array(eps).reshape(1)
        self.policy = self.update_policy()

    def take_action(self, policy):
        action = policy.rvs()
        self.action_counts[action] += 1
        return action

    def get_reward(self, action):
        reward_loc = self.reward_mean[action]
        reward = st.norm.rvs(loc=reward_loc)
        return reward

    def update_policy(self, ):
        argmax = np.argmax(self.reward_estimates)
        pmf = np.ones(self.num_actions)
        pmf = pmf / sum(pmf) * (self.num_actions / (self.num_actions - 1)) * self.eps
        pmf[argmax] = 1 - self.eps
        self.policy = st.rv_discrete(values=(self.actions, pmf))
        return self.policy

    def change_reward(self):
        self.reward_mean += st.norm.rvs(size=10)


samples = 10000
eps = 0.1
repeats = 100
run_average = []
optimal_percent = []
for repeat in range(repeats):
    bandits = KBandits(10, eps)
    rewards = []
    optimal_action_count = 0
    for i in range(1, samples + 1):
        action = bandits.take_action(bandits.policy)
        if action == np.argmax(bandits.reward_mean):
            optimal_action_count += 1
        reward = bandits.get_reward(action)
        rewards.append(reward)
        bandits.reward_estimates[action] = bandits.reward_estimates[action] + (
                reward - bandits.reward_estimates[action]) / \
                                           bandits.action_counts[action]
        bandits.update_policy()
    run_average.append(sum(rewards) / samples)
    optimal_percent.append(optimal_action_count / samples)
    print(f"repeat:{repeat},reward:{run_average[repeat]},optimal percent:{optimal_percent[repeat]}")
    '''
    ps = []
    for i in range(bandits.num_actions):
        ps.append(bandits.policy.pmf(i))
    '''

xs = np.arange(1, repeats + 1)
plt.subplot(1, 2, 1)
plt.plot(xs, np.array(run_average))
plt.subplot(1, 2, 2)
plt.plot(xs, np.array(optimal_percent))
plt.show()
'''print(
    f"Reward mean:\n{bandits.reward_mean}\nReward estimates:\n{bandits.reward_estimates}\nAction pmf:\n{np.array(ps)}\n"
    f"Action counts:\n{bandits.action_counts},\noptimal action count:{optimal_action_count}")
'''

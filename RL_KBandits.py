import numpy as np
import scipy.stats as st
np.set_printoptions(precision=4)

class KBandits:
    def __init__(self, number_of_bandits, eps):
        self.num_actions = number_of_bandits
        self.reward_mean = st.norm(loc=0, scale=1).rvs(size=self.num_actions).reshape(10)
        self.actions = np.arange(0, self.num_actions).astype(np.float64).reshape(10)
        self.action_counts = np.zeros(self.num_actions)
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


samples = 10000
bandits = KBandits(10, 0.5)
for i in range(samples):
    action = bandits.take_action(bandits.policy)
    reward = bandits.get_reward(action)
    bandits.reward_estimates[action] = bandits.reward_estimates[action] + (reward - bandits.reward_estimates[action]) / \
                                       bandits.action_counts[action]
    bandits.update_policy()
ps = []
for i in range(bandits.num_actions):
    ps.append(bandits.policy.pmf(i))
print(
    f"Reward mean:\n{bandits.reward_mean}\nReward estimates:\n{bandits.reward_estimates}\nAction pmf:\n{np.array(ps)}\nAction counts:\n{bandits.action_counts}")

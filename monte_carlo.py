"""
Monte Carlo estimation on 4x4 grid world
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


class GridWorld:
    def __init__(self):
        self.action_space = np.arange(0, 4)
        self.environment = np.arange(0, 16).reshape(-1, 4)
        self.initial_state = None
        self.current_state = None
        self.episode_ended = False
        self.reward_distribution = st.randint(-1, 0)

    def sample_action(self):
        action = st.randint(np.min(self.action_space), np.max(self.action_space) + 1).rvs(1)
        return action

    def take_action(self, action):
        if self.episode_ended:
            return self.current_state, 0
        [[row, col]] = np.argwhere(self.environment == self.current_state)
        reward = self.reward_distribution.rvs(1)
        if action == 0:
            if row - 1 >= 0:
                self.current_state -= 4
            else:
                return self.current_state, reward
        if action == 1:
            if row + 1 <= 3:
                self.current_state += 4
            else:
                return self.current_state, reward
        if action == 2:
            if col - 1 >= 0:
                self.current_state -= 1
            else:
                return self.current_state, reward
        if action == 3:
            if col + 1 <= 3:
                self.current_state += 1
            else:
                return self.current_state, reward
        if self.current_state == 0 or self.current_state == 15:
            self.episode_ended = True
            return self.current_state, reward
        return self.current_state, reward

    def init_env(self):
        self.initial_state = st.randint(np.min(self.environment) + 1, np.max(self.environment)).rvs(1)
        self.current_state = self.initial_state
        return self.initial_state


class Agent:
    def __init__(self, environment, num_states=16, num_actions=4):
        self.experience = []
        self.grid_world = np.arange(0, 16).reshape(-1, 4)
        self.policy = np.ones((num_states, num_actions)) / num_actions
        self.state_value_estimates = np.zeros(num_states)
        self.state_update_count = np.zeros(num_states)
        self.legal_moves = self.get_legal_moves()
        self.env = environment

    def get_episode(self, env):
        cs = env.current_state
        trajectory = [*cs]
        rewards = [0]
        while not env.episode_ended:
            action_pmf = st.rv_discrete(
                values=(env.action_space.reshape(-1), self.policy[cs].reshape(-1)))
            action = action_pmf.rvs(size=1)
            cs, rwd = env.take_action(action)
            trajectory.append(*cs)
            rewards.append(*rwd)
        return np.array(trajectory), np.array(rewards)

    def learning_with_monte_carlo(self):
        gamma = np.array(1)
        env = GridWorld()
        env.init_env()
        traj, rwds = self.get_episode(env)
        for i in range(1, 15):
            if len(np.argwhere(traj == i)) != 0:
                first_occurance, = np.argwhere(traj == i)[0]
                subtrajectory = traj[first_occurance:]
                subtrajectory_length = len(subtrajectory)
                powers = np.power(gamma, np.arange(0, subtrajectory_length))
                cum_return = np.sum(rwds[first_occurance:] * powers)
                self.state_value_estimates[i] = (self.state_value_estimates[i] * self.state_update_count[
                    i] + cum_return) / (self.state_update_count[i] + 1)
                self.state_update_count[i] += 1
        self.update_policy()

    def update_policy(self, eps=0.01):
        for j in range(1, 15):
            reachable_state_value = self.state_value_estimates[self.legal_moves[j, :]]
            optimal_action = np.argmax(reachable_state_value)
            self.policy[j, :] = np.ones((1, 4)) * eps / 3
            self.policy[j, optimal_action] = 1 - eps

    def get_legal_moves(self):
        legal_moves = np.zeros((1, 4), dtype=np.int32)
        for s in range(1, 16):
            legal_moves = np.concatenate((legal_moves, np.ones((1, 4)) * s), axis=0)
        for i in range(1, 15):
            [[row, col]] = np.argwhere(self.grid_world == i)
            if row - 1 >= 0:
                legal_moves[i, 0] = i - 4
            if row + 1 <= 3:
                legal_moves[i, 1] = i + 4
            if col - 1 >= 0:
                legal_moves[i, 2] = i - 1
            if col + 1 <= 3:
                legal_moves[i, 3] = i + 1
        legal_moves = np.array(legal_moves, dtype=np.int32)
        return legal_moves


class Canvas:
    def __init__(self, shape):
        self.fig, self.axes = plt.subplots(1, 2, layout="constrained")
        initial_values = np.arange(16).reshape(shape).transpose()
        '''set state values axes'''
        self.axes[0].axis("off")
        self.axes[0].set_title("State Values Estimates")
        self.axes[0].xaxis.tick_top()
        self.axes[0].invert_yaxis()
        self.axes[0].matshow(initial_values, cmap="Blues")

        '''set policy axes'''
        self.axes[1].set_title("Policy")
        self.axes[1].axis("off")
        self.axes[1].xaxis.tick_top()
        self.axes[1].invert_yaxis()
        self.axes[1].matshow(initial_values, cmap="Blues")
        self.arrow_head_width = 0.1
        self.arrow_head_length = 0.15
        self.axes[1].scatter(0, 0, s=80, c="red", marker="o")
        self.axes[1].scatter(3, 3, s=80, c="red", marker="o")
        for (i, j), v in np.ndenumerate(initial_values):
            self.axes[0].text(i, j, str(v))
            if not ((i == 0 and j == 0) or (i == 3 and j == 3)):
                self.axes[1].arrow(i, j, 0.1, 0, shape="full", head_width=self.arrow_head_width, color="green",
                                   length_includes_head=False)
                self.axes[1].arrow(i, j, 0, 0.1, shape="full", head_width=self.arrow_head_width, color="green",
                                   length_includes_head=False)
                self.axes[1].arrow(i, j, -0.1, 0, shape="full", head_width=self.arrow_head_width, color="green",
                                   length_includes_head=False)
                self.axes[1].arrow(i, j, 0, -0.1, shape="full", head_width=self.arrow_head_width, color="green",
                                   length_includes_head=False)

    def repaint(self, values, policy):
        self.axes[0].clear()
        self.axes[0].set_title("State Values Estimates")
        self.axes[0].axis("off")
        self.axes[0].matshow(values, cmap="Blues")
        for (i, j), v in np.ndenumerate(values):
            self.axes[0].text(i, j, str(v))
        self.axes[1].clear()
        self.axes[1].set_title("Policy")
        self.axes[1].axis("off")
        self.axes[1].xaxis.tick_top()
        self.axes[1].matshow(values, cmap="Blues")
        self.axes[1].scatter(0, 0, s=80, c="red", marker="o")
        self.axes[1].scatter(3, 3, s=80, c="red", marker="o")
        for (i, j), v in np.ndenumerate(policy):
            if not ((i == 0 and j == 0) or (i == 3 and j == 3)):
                if v == 0:
                    self.axes[1].arrow(i, j, 0, -0.1, shape="full", head_width=self.arrow_head_width,
                                       head_length=self.arrow_head_length, color="green", length_includes_head=False)
                if v == 1:
                    self.axes[1].arrow(i, j, 0, 0.1, shape="full", head_width=self.arrow_head_width,
                                       head_length=self.arrow_head_length, color="green", length_includes_head=False)
                if v == 2:
                    self.axes[1].arrow(i, j, -0.1, 0, shape="full", head_width=self.arrow_head_width,
                                       head_length=self.arrow_head_length, color="green", length_includes_head=False)
                if v == 3:
                    self.axes[1].arrow(i, j, 0, -0.1, shape="full", head_width=self.arrow_head_width,
                                       head_length=self.arrow_head_length, color="green", length_includes_head=False)


gw = GridWorld()
gw.init_env()
agent = Agent(gw)
plt.ion()
can = Canvas((4, 4))
plt.figure(can.fig)
plt.show()
plt.pause(2)
for i in range(20000):
    if i % 25 == 0 and i != 0:
        values = np.round(agent.state_value_estimates.reshape(-1, 4), decimals=1).transpose()
        policy = np.argmax(agent.policy, axis=1).reshape(-1, 4).transpose()
        can.repaint(values, policy)
        plt.pause(0.5)
    agent.learning_with_monte_carlo()

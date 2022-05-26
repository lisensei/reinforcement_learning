"""
Monte Carlo value iteration on 4x4 grid world
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from collections import namedtuple
import matplotlib as mpl

'''This class creates the grid world environment'''
transition = namedtuple("transition", ["state", "action", "next_state", "reward"])


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

    def reset(self):
        self.initial_state = st.randint(np.min(self.environment) + 1, np.max(self.environment)).rvs(1)
        self.current_state = self.initial_state
        self.episode_ended = False
        return self.current_state


'''This class creates an agent'''


class Agent:
    def __init__(self, env, num_states=16, num_actions=4, gamma=1):
        self.experience = []
        self.environment = env
        self.grid_world = np.arange(0, 16).reshape(-1, 4)
        self.policy = np.ones((num_states, num_actions)) / num_actions
        self.state_value_estimates = np.zeros(num_states)
        self.action_value_estimates = np.zeros((num_states, num_actions))
        self.state_update_count = np.zeros(num_states)
        self.action_value_update_count = np.zeros((num_states, num_actions))
        self.legal_moves = self.get_legal_moves()
        self.policy_update_count = 0
        self.gamma = gamma

    def get_episode(self):
        self.environment.reset()
        episode = []
        episode_length = 0
        while not self.environment.episode_ended:
            current_state = np.copy(self.environment.current_state)
            action_pmf = st.rv_discrete(
                values=(self.environment.action_space.reshape(-1), self.policy[current_state].reshape(-1)))
            action = action_pmf.rvs(size=1)
            next_state, reward = self.environment.take_action(action)
            self.append = episode.append(transition(current_state, action, np.copy(next_state), reward))
            episode_length += 1
            if episode_length % 100 == 0:
                print(f"episode length gets large: {episode_length}")
        return episode

    def update_value_estimates(self, transitions):
        env = GridWorld()
        env.init_env()
        episode_length = len(transitions)
        episode = np.array(transitions).reshape(-1, 4)
        trajectory, rewards = episode[:, 0].reshape(-1), episode[:, 3].reshape(-1)
        '''Computes state values'''
        for i in range(1, 15):
            if len(np.argwhere(trajectory == i)) != 0:
                first_occurrence, = np.argwhere(trajectory == i)[0]
                subtrajectory = trajectory[first_occurrence:]
                subtrajectory_length = len(subtrajectory)
                powers = np.power(self.gamma, np.arange(0, subtrajectory_length))
                cum_return = np.sum(rewards[first_occurrence:] * powers)
                self.state_value_estimates[i] = (self.state_value_estimates[i] * self.state_update_count[
                    i] + cum_return) / (self.state_update_count[i] + 1)
                self.state_update_count[i] += 1
        whether_updated = dict()
        for i, t in enumerate(episode):
            s, a = t[0], t[1]
            key = f"{s}-{a}"
            if key in whether_updated:
                # print(f"iteration:{i},state-action {s}-{a} already updated: continued")
                continue
            whether_updated[key] = 1
            action_value = np.sum(rewards[i:] * self.gamma ** np.arange(
                episode_length - i))
            self.action_value_estimates[s, a] = (self.action_value_estimates[s, a] * \
                                                 self.action_value_update_count[s, a] + action_value) / (
                                                        self.action_value_update_count[s, a] + 1)
            self.action_value_update_count[s, a] += 1
        '''
        state_values_from_action_values = np.sum(
            self.action_value_estimates * self.policy, axis=1)
        self.state_value_estimates = state_values_from_action_values
        print(f"state values:\n{np.round(self.state_value_estimates, decimals=2).reshape(4, 4)}")
        print(f"from action values:\n{np.round(state_values.reshape(4, 4), decimals=2)}")
        '''

    '''Epsilon greedy and set equal probability for equal state values to avoid infinite loop'''

    def update_policy(self, eps=0.1):
        eps = eps ** (1 + self.policy_update_count / 1000) if eps != 0 else eps
        for j in range(1, 15):
            reachable_state_value = self.state_value_estimates[self.legal_moves[j, :]]
            max_index = np.argmax(reachable_state_value)
            optimal_action = np.argwhere(reachable_state_value == reachable_state_value[max_index])
            num_optimal_actions = len(optimal_action)
            if num_optimal_actions < 4:
                self.policy[j, :] = np.ones((1, 4)) * eps / (4 - num_optimal_actions)
                self.policy[j, optimal_action] = (1 - eps) / num_optimal_actions
            else:
                self.policy[j, :] = np.ones((1, 4)) / 4
        self.policy_update_count += 1

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


'''This class is used to visualize state values and policy'''


class Canvas:
    def __init__(self, shape):
        self.fig, self.axes = plt.subplot_mosaic([["state values", "policy"], ["action values", "action values"]],
                                                 figsize=(14, 12),
                                                 layout="constrained")
        initial_values = np.zeros(16).reshape(shape).transpose()
        initial_action_values = np.zeros((16, 4)).transpose()
        '''Set state values axes'''
        self.axes["state values"].invert_yaxis()
        self.set_state_value_plot(initial_values)

        '''Set policy axes'''
        self.axes["policy"].invert_yaxis()
        self.set_policy_plot(initial_values, initial_values)
        for (i, j), v in np.ndenumerate(initial_values):
            if not ((i == 0 and j == 0) or (i == 3 and j == 3)):
                self.axes["policy"].arrow(i, j, 0.1, 0, shape="full", head_width=self.arrow_head_width, color="green",
                                          length_includes_head=False)
                self.axes["policy"].arrow(i, j, 0, 0.1, shape="full", head_width=self.arrow_head_width, color="green",
                                          length_includes_head=False)
                self.axes["policy"].arrow(i, j, -0.1, 0, shape="full", head_width=self.arrow_head_width, color="green",
                                          length_includes_head=False)
                self.axes["policy"].arrow(i, j, 0, -0.1, shape="full", head_width=self.arrow_head_width, color="green",
                                          length_includes_head=False)
        '''Set action value axes'''
        self.axes["action values"].invert_yaxis()
        self.set_action_value_plot(initial_action_values)

    '''Plot state values and policy. Only one policy is plotted'''

    def set_state_value_plot(self, state_values):
        self.axes["state values"].axis("off")
        self.axes["state values"].set_title("State Values Estimates")
        self.axes["state values"].matshow(state_values, cmap="Blues")
        for (i, j), v in np.ndenumerate(state_values):
            self.axes["state values"].text(i, j, str(v))

    def set_policy_plot(self, state_values, policy):
        self.axes["policy"].set_title("Policy")
        self.axes["policy"].axis("off")
        self.axes["policy"].xaxis.tick_top()
        self.axes["policy"].matshow(state_values, cmap="Blues")
        self.arrow_head_width = 0.1
        self.arrow_head_length = 0.15
        self.axes["policy"].scatter(0, 0, s=80, c="red", marker="o")
        self.axes["policy"].scatter(3, 3, s=80, c="red", marker="o")
        for (i, j), v in np.ndenumerate(policy):
            if not ((i == 0 and j == 0) or (i == 3 and j == 3)):
                if v == 0:
                    self.axes["policy"].arrow(i, j, 0, -0.1, shape="full", head_width=self.arrow_head_width,
                                              head_length=self.arrow_head_length, color="green",
                                              length_includes_head=False)
                if v == 1:
                    self.axes["policy"].arrow(i, j, 0, 0.1, shape="full", head_width=self.arrow_head_width,
                                              head_length=self.arrow_head_length, color="green",
                                              length_includes_head=False)
                if v == 2:
                    self.axes["policy"].arrow(i, j, -0.1, 0, shape="full", head_width=self.arrow_head_width,
                                              head_length=self.arrow_head_length, color="green",
                                              length_includes_head=False)
                if v == 3:
                    self.axes["policy"].arrow(i, j, 0.1, 0, shape="full", head_width=self.arrow_head_width,
                                              head_length=self.arrow_head_length, color="green",
                                              length_includes_head=False)

    def set_action_value_plot(self, action_values):
        self.axes["action values"].set_title("Action values Estimates", )
        self.axes["action values"].matshow(action_values, cmap="plasma")
        self.axes["action values"].xaxis.tick_bottom()
        self.axes["action values"].xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        self.axes["action values"].set_xlabel("State")
        self.axes["action values"].set_ylabel("Action")
        self.axes["action values"].tick_params(bottom=None, top=None, left=None, right=None)
        self.axes["action values"].set_yticks(np.arange(4), ["Up", "Down", "Left", "Right"])
        for (i, j), v in np.ndenumerate(action_values):
            self.axes["action values"].text(j, i, str(v))

    def repaint(self, values, policy, action_values):
        '''State values plotting'''
        self.axes["state values"].clear()
        self.set_state_value_plot(values)

        '''Visualizing policy'''
        self.axes["policy"].clear()
        self.set_policy_plot(values, policy)

        '''Plot action values'''
        transposed_action_values = np.round(action_values.transpose(), decimals=1)
        self.axes["action values"].clear()
        self.set_action_value_plot(transposed_action_values)


gw = GridWorld()
agent = Agent(gw)
plt.ion()
can = Canvas((4, 4))
plt.figure(can.fig)
plt.show()
plt.pause(2)
for e in range(20000):
    if e % 25 == 0 and e != 0:
        values = np.round(agent.state_value_estimates.reshape(-1, 4), decimals=1).transpose()
        policy = np.argmax(agent.policy, axis=1).reshape(-1, 4).transpose()
        can.repaint(values, policy, agent.action_value_estimates)
        plt.pause(0.5)
    agent.update_value_estimates(agent.get_episode())
    agent.update_policy()

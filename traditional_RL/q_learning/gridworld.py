import numpy as np
import scipy.stats as st

'''This class creates the grid world environment'''


class GridWorld:
    def __init__(self, num_states=16, num_actions=4):
        self.num_states = num_states
        self.shape = (4, 4)
        self.num_actions = num_actions
        self.action_space = np.arange(0, num_actions)
        self.environment = np.arange(0, num_states).reshape(self.shape)
        self.initial_state = None
        self.current_state = None
        self.episode_ended = False
        self.reward_distribution = st.randint(-1, 0)

    def sample_action(self):
        action = st.randint(np.min(self.action_space), np.max(self.action_space) + 1).rvs(1)
        return action

    def take_action(self, action):
        if self.episode_ended:
            return self.current_state, 0, 1
        [[row, col]] = np.argwhere(self.environment == self.current_state)
        reward = self.reward_distribution.rvs(1)
        if action == 0:
            if row - 1 >= 0:
                self.current_state -= 4
            else:
                return self.current_state, reward, 0
        if action == 1:
            if row + 1 <= 3:
                self.current_state += 4
            else:
                return self.current_state, reward, 0
        if action == 2:
            if col - 1 >= 0:
                self.current_state -= 1
            else:
                return self.current_state, reward, 0
        if action == 3:
            if col + 1 <= 3:
                self.current_state += 1
            else:
                return self.current_state, reward, 0
        if self.current_state == 0 or self.current_state == 15:
            self.episode_ended = True
            return self.current_state, reward, 1
        return self.current_state, reward, 0

    def init_env(self):
        self.initial_state = st.randint(np.min(self.environment) + 1, np.max(self.environment)).rvs(1)
        self.current_state = self.initial_state
        return self.initial_state

    def reset(self):
        self.initial_state = st.randint(np.min(self.environment) + 1, np.max(self.environment)).rvs(1)
        self.current_state = self.initial_state
        self.episode_ended = False
        return self.current_state

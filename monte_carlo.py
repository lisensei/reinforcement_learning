import random

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


class GridWorld:
    def __init__(self):
        self.action_space = np.arange(0, 4)
        self.environment = np.arange(0, 16).reshape(-1, 4)
        self.initial_state = None
        self.current_state = None

    def sample_action(self):
        action = st.randint(np.min(self.action_space), np.max(self.action_space) + 1).rvs(1)
        return action

    def take_action(self, action):
        [[row, col]] = np.argwhere(self.environment == self.current_state)
        if self.current_state == 0 or self.current_state == 15:
            print(f"Terminal State Reached:{self.current_state}")
            return self.current_state
        if action == 0:
            if row - 1 >= 0:
                self.current_state -= 4
            else:
                return self.current_state
        if action == 1:
            if row + 1 <= 3:
                self.current_state += 4
            else:
                return self.current_state
        if action == 2:
            if col - 1 >= 0:
                self.current_state -= 1
            else:
                return self.current_state
        if action == 3:
            if col + 1 <= 3:
                self.current_state += 1
            else:
                return self.current_state
        return self.current_state

    def init_env(self):
        self.initial_state = st.randint(np.min(self.environment), np.max(self.environment) + 1).rvs(1)
        self.current_state = self.initial_state
        return self.initial_state


gw = GridWorld()
ini = gw.init_env()
for i in range(1000):
    a = np.random.randint(0, 4)
    state = gw.take_action(a)
    print(state)

import numpy
import numpy as np
from collections import *

np.set_printoptions(precision=4)
num_grid = 16
num_actions = 4
policy = np.ones((num_grid, num_actions)) * 0.25
grid_world = np.arange(num_grid).reshape(-1, int(num_grid ** 0.5))
state = namedtuple("state", ["state", "next_states", "prob"])


def generate_states():
    states = []
    for (row, col), value in np.ndenumerate(grid_world):
        next_states = np.ones(4, dtype=int) * value
        prob = np.ones(4) * 0.25
        if row - 1 >= 0:
            next_states[0] = value - 4
        if row + 1 <= 3:
            next_states[1] = value + 4
        if col - 1 >= 0:
            next_states[2] = value - 1
        if col + 1 <= 3:
            next_states[3] = value + 1
        states.append(state(value, next_states, prob))
    return states


def evaluate_policy(policy, states):
    state_values = np.zeros(num_grid)
    num_state = len(state_values)
    eps = 1e-5
    temp = np.ones(num_state)
    while np.abs(np.sum(temp - state_values)) > eps:
        temp = numpy.copy(state_values)
        for i in range(1, num_state - 1):
            new_value = state_values[states[i].next_states] @ policy[i].transpose() - 1
            state_values[i] = new_value
    return state_values


def update_policy(policy):
    return policy


dic = {0: "up", 1: "down", 2: "left", 3: "right"}
states_array = generate_states()

new = evaluate_policy(policy, states_array)
print(new.reshape(-1, 4))

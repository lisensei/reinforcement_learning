"""
policy iteration algorithm applied on a 4x4 grid world with state 0 and 15 being the terminal states
"""
import numpy
import numpy as np
from collections import *

np.set_printoptions(precision=4)
num_grid = 16
num_actions = 4
policy = np.ones((num_grid, num_actions)) * 0.25
grid_world = np.arange(num_grid).reshape(-1, int(num_grid ** 0.5))
state = namedtuple("state", ["state", "next_states", "prob"])

'''
This function generates legal moves from a given state
'''


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


'''
This function evaluates a given policy and returns the expected value of being in a given state.
'''


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


'''
This function improves policy by trying out different actions
'''


def update_policy(states, policy, state_values):
    num_states = len(states)
    num_actions = policy.shape[1]
    for i in range(1, num_states - 1):
        next_states = states[i].next_states
        try_out_actions = np.ones(num_actions)
        action_values = state_values[next_states] * try_out_actions - 1
        optimal_action = np.argmax(action_values)
        new_policy = np.zeros(num_actions)
        new_policy[optimal_action] = 1
        policy[i, :] = new_policy
    return policy


dic = {0: "up  ", 1: "down", 2: "left", 3: "right"}
states_array = generate_states()

for it in range(100):
    new_state_values = evaluate_policy(policy, states_array)

    policy = update_policy(states_array, policy, new_state_values)

optimal_state_values = new_state_values.reshape(-1, 4)
optimal_policy = np.argmax(policy, axis=1)

print(f"optimal state values:")
print(optimal_state_values)
print(f"\none optimal policy:")
for i in range(len(optimal_policy)):
    if i % 4 == 0 and i != 0:
        print("\n", end="")
    if i == 0 or i == 15:
        print("end ", "\t", end="")
    else:
        print(dic[optimal_policy[i]], "\t",
              end="")

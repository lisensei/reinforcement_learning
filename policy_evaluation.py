'''Uniform policy evaluation on 4x4 grid world'''

from itertools import *
import numpy as np

'''Make a grid world'''
grid = np.arange(16).reshape(4, 4)

'''Non-terniminal states and actions'''
nonterminal_state = range(15)
actions = ["up", "down", "left", "right"]

'''Meaning of (x,y,z) tuple: x represents the operation on the row 0, 1 represents operation on column;
   y represents the direction of the operation. If (x,y)=(0,-1) it means take the action "up";if (x,y)=(1,1)
   it means take the action "right". 
   z is used to compute the new state number. If an "up" action is legal, for example going up from state number 5, it means the
   new state number is the 5 subtracted by 4.     
'''
values = [(0, -1, -4), (0, 1, 4), (1, -1, -1), (1, 1, 1)]
lookup = {action: value for action, value in zip(actions, values)}

'''Make all legal state-action-next state-reward tuples'''
state_action_pair = [list(x) for x in product(nonterminal_state, actions)]
next_state = [[x[0] + lookup[x[1]][2], -1] if 0 <= np.argwhere(grid == x[0])[0][lookup[x[1]][0]] +
                                              lookup[x[1]][1] <= 3 else [x[0], -1] for x in state_action_pair]
sansr = [[x, y, z, w] for [x, y], [z, w] in zip(state_action_pair, next_state)]



def update(state_values, policy, state_action_nextstate_reward):
    for i in range(1, 15):
        tuples = [x for x in state_action_nextstate_reward if x[0] == i]
        g = 0
        for tuple in tuples:
            g += policy[tuple[1]] * (state_values[tuple[2]] + tuple[3])

        state_values[i] = g


state_values = np.zeros(16)
policy = {action: 0.25 for action in actions}
for t in range(1000):
    update(state_values, policy, sansr)

state_values = np.array(state_values).reshape(grid.shape)
print(state_values)

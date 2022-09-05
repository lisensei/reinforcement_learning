import time
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from gridworld import *
from collections import namedtuple, deque

transition = namedtuple("transition", ["state", "action", "next_state", "reward"])

state_space_size = 16
action_space_size = 4


class Agent:
    def __init__(self, env, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.q_values = np.zeros((state_size, action_size))
        self.memory = deque(maxlen=10000)
        pass

    def q_learning(self, num_episodes=10, learning_rate=0.1):
        '''

        :param env:
        :param num_episodes:
        :param learning_rate:
        :return: None

        Stationary:
        Q(s,a)  = (Q(s,a)*n + Q'(s,a))/(n + 1)
                = Q(s,a) + (Q'(s,a) - Q(s,a))/(n + 1)
        None stationary
        Q(s,a)  =  Q(s,a) + a(Q'(s,a) - Q(s,a))
                =  Q(s,a) + a(R_{t+1} + Q_max(s',a')-Q(s,a))

        '''
        fig, axe = plt.subplots(layout="constrained")
        axe.yaxis_inverted()
        for e in range(num_episodes):
            episode = self.sample_episode(self.env)
            num_transitions = len(episode)
            for i, t in enumerate(episode):
                state, action, reward, next_state = t.state, t.action, t.reward, t.next_state
                q_of_sa = self.q_values[state, action]
                if i < num_transitions - 1:
                    td_target = reward + np.max(self.q_values[next_state])
                else:
                    td_target = reward
                new_q_of_sa = q_of_sa + learning_rate * (td_target - q_of_sa)
                self.q_values[state, action] = new_q_of_sa
            if e % 100 == 0:
                print(self.q_values)
                self.show_qvalues(axe)

    def sample_action(self, state, eps=0.4):
        eps_over_A = (eps / self.action_size)
        action_given_state_probabilities = np.ones((self.action_size)) * eps_over_A
        argmax_a = int(np.max(self.q_values[state]))
        action_given_state_probabilities[argmax_a] = 1 - eps + eps_over_A
        action_sampler = st.rv_discrete(values=(np.arange(self.action_size), action_given_state_probabilities))
        action = np.array(action_sampler.rvs())
        return action

    def sample_episode(self, env: GridWorld):
        state = env.reset()
        done = False
        episode = []
        while not done:
            state_copy = state.copy()
            action = self.sample_action(state)
            state, reward, done = env.take_action(action)
            t = transition(state_copy, action, state.copy(), reward)
            episode.append(t)
        return episode

    def show_qvalues(self, axe):
        transposed_matrix = self.q_values.transpose().round(1)
        axe.clear()
        axe.matshow(transposed_matrix)
        for (i, j), v in np.ndenumerate(transposed_matrix):
            axe.text(j, i, str(v))
            plt.show()
        plt.pause(0.1)

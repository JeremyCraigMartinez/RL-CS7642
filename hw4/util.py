'''utility functions'''

from sys import argv
import numpy as np

def logger(_str, level='low'):
    '''logger enabled with --verbose flag'''
    if '--verbose' in argv or level == 'high':
        print(_str)

# pylint: disable=invalid-name
def select_epsilon_greedy_action(_state, _Q, epsilon, env):
    '''epsilon greedy action selection'''
    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n)
    return np.argmax(_Q[_state, :])

# pylint: disable=invalid-name
def select_greedy_action(_state, _Q):
    '''epsilon greedy action selection'''
    return np.argmax(_Q[_state, :])

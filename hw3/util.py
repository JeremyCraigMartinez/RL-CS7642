'''utility functions'''

from math import sqrt
from sys import argv
import numpy as np

def square(amap):
    '''square amap input from string to square array'''
    # pylint: disable=line-too-long
    return [amap[i * int(sqrt(len(amap))) : (i + 1) * int(sqrt(len(amap)))] for i in range(int(sqrt(len(amap))))]

def logger(_str, level='low'):
    '''logger enabled with --verbose flag'''
    if '--verbose' in argv or level == 'high':
        print(_str)

def map_action_list(action_list_array):
    '''Map action list of numbers (01023222301) to accepted string of arrows'''
    action_list_str = ''.join([str(action) for action in action_list_array])
    action_list = ''
    for char in action_list_str:
        if char == '0':
            action_list = '{}{}'.format(action_list, '<')
        elif char == '1':
            action_list = '{}{}'.format(action_list, 'v')
        elif char == '2':
            action_list = '{}{}'.format(action_list, '>')
        else:
            action_list = '{}{}'.format(action_list, '^')
    return action_list

# pylint: disable=invalid-name
def select_action(_state, _Q, epsilon, env):
    '''epsilon greedy action selection'''
    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n)
    return np.argmax(_Q[_state, :])

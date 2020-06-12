'''utility functions'''

from math import sqrt
from sys import argv

def square_amap(amap):
    '''square amap input from string to square array'''
    # pylint: disable=line-too-long
    return [amap[i * int(sqrt(len(amap))) : (i + 1) * int(sqrt(len(amap)))] for i in range(int(sqrt(len(amap))))]

def logger(_str):
    '''logger enabled with --verbose flag'''
    if '--verbose' in argv:
        print(_str)

def map_action_list(_str):
    '''Map action list of numbers (01023222301) to accepted string of arrows'''
    action_list = ''
    for char in _str:
        if char == '0':
            action_list = '{}{}'.format(action_list, '<')
        elif char == '1':
            action_list = '{}{}'.format(action_list, 'v')
        elif char == '2':
            action_list = '{}{}'.format(action_list, '>')
        else:
            action_list = '{}{}'.format(action_list, '^')

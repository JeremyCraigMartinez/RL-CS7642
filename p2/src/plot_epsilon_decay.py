'''Test the ep decay'''

import util

def ep_decay(epsilon, decay):
    '''Test epsilon decay lifespan'''
    epsilons = []
    for _ in range(1000):
        epsilon *= (1 - decay)
        epsilons.append(epsilon)
    return epsilons

list_of_epsilon_decays = [0.003, 0.005, 0.007]
decay_arr = [ep_decay(1, decay) for decay in list_of_epsilon_decays]
# pylint: disable=line-too-long
util.plot_multiple(decay_arr, list_of_epsilon_decays, 'Episodes', 'Epsilon Value', 'epsilon-decay.png')

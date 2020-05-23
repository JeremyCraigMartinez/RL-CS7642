'''CS 7642 - HW1 Solution'''

from sys import argv
import mdptoolbox.example
import numpy as np

def main(is_bad_side):
    '''Main function for solution'''
    run = 3
    length = len(is_bad_side)
    num_states = run * length + 2
    is_good_side = ~is_bad_side+2
    dollar = np.arange(1, length + 1) * is_good_side
    prob = np.zeros((2, num_states, num_states))
    np.fill_diagonal(prob[0], 1)
    p = 1.0 / length
    zero = np.array([0]).repeat((run - 1) * length + 2)

    def summation(arr1, arr2):
        for _ in range(0, run * length + 2):
            arr1 = np.concatenate((arr1, arr2), axis=0)
        return arr1

    # 1
    is_good_side_2 = np.concatenate((np.array([0]), is_good_side, zero), axis=0)
    is_good_side_n = np.concatenate((is_good_side_2, is_good_side_2), axis=0)
    is_good_side_n = summation(is_good_side_n, is_good_side_2)
    is_good_side_n = is_good_side_n[:(num_states ** 2)]
    is_good_side_n = is_good_side_n.reshape(num_states, num_states)
    prob[1] = np.triu(is_good_side_n)
    prob[1] = prob[1]*p
    prob_end = 1 - np.sum(prob[1, :num_states, :num_states-1], axis=1).reshape(-1, 1)

    prob[1] = np.concatenate((prob[1, :num_states, :num_states-1], prob_end), axis=1)
    np.sum(prob[0], axis=1)
    np.sum(prob[1], axis=1)
    rewards = np.zeros((2, num_states, num_states))
    rewards[0] = np.zeros((num_states, num_states))

    # 2
    dollar_2 = np.concatenate((np.array([0]), dollar, zero), axis=0)
    dollar_n = np.concatenate((dollar_2, dollar_2), axis=0)
    dollar_n = summation(dollar_n, dollar_2)
    dollar_n = dollar_n[:(num_states ** 2)]
    dollar_n = dollar_n.reshape(num_states, num_states)
    rewards[1] = np.triu(dollar_n)
    rewards_end = - np.array(range(0, num_states)).reshape(-1, 1)
    rewards[1] = np.concatenate((rewards[1, :num_states, :num_states-1], rewards_end), axis=1)

    val_it = mdptoolbox.mdp.ValueIteration(prob, rewards, 1)
    val_it.run()

    # optimal_policy = val_it.policy
    expected_values = val_it.V

    print('Answer is: {}'.format(max(expected_values)))

if __name__ == "__main__":
    EXAMPLE_1 = np.array([1, 1, 1, 0, 0, 0])
    EXAMPLE_2 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0])
    EXAMPLE_3 = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0])

    if '--help' in argv:
        print('''
NAME
     HW1 solution

SYNOPSIS
     python hw1 [--example] [--help]

DESCRIPTION
     This script is the solution for homework 1 of CS 7642 - Reinforcement learning. You can either run an example solution from the homework description, or you can edit this script to test new problems.

     The following options are available:

     --example
             Pass in one of the example problems (i.e. --example 2)
        ''')
        exit()

    if '--example' in argv:
        if argv[len(argv) - 1] == '1':
            main(EXAMPLE_1)
        elif argv[len(argv) - 1] == '2':
            main(EXAMPLE_2)
        elif argv[len(argv) - 1] == '3':
            main(EXAMPLE_3)
    if '--submission' in argv:
        HOMEWORK_VALUES = [
            np.array([1, 1, 1, 0, 0, 0]),
            np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]),
            np.array([1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
        ]
        for _ in HOMEWORK_VALUES:
            main(_)
    else:
        # EDIT LINE BELOW TO TEST NEW ARRAY
        IS_BAD_SIDE = []
        main(IS_BAD_SIDE)

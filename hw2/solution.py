'''HW2 Solution'''

from scipy.optimize import fsolve
import numpy as np

# pylint: disable=invalid-name
class TD_lambda:
    '''TD(lambda) class'''
    def __init__(self, prob_to_state, val_estimates, rewards):
        '''Initialize variables in TD_lambda class'''
        self.prob_to_state = prob_to_state
        self.val_estimates = val_estimates
        self.rewards = rewards

    def get_v_s0(self, _lambda):
        '''Get value of initial state in MDP'''

        # pylint: disable=line-too-long
        v_s0 = self.val_estimates[0] + \
            self.prob_to_state * (
                _lambda ** 0 * self.rewards[0] + \
                _lambda ** 1 * self.rewards[2] + \
                _lambda ** 2 * self.rewards[4] + \
                _lambda ** 3 * self.rewards[5] + \
                _lambda ** 4 * self.rewards[6] + \
                _lambda ** 0 * (1 - _lambda) * self.val_estimates[1] + \
                _lambda ** 1 * (1 - _lambda) * self.val_estimates[3] + \
                _lambda ** 2 * (1 - _lambda) * self.val_estimates[4] + \
                _lambda ** 3 * (1 - _lambda) * self.val_estimates[5] + \
                _lambda ** 4 * (1 - _lambda) * self.val_estimates[6] - \
                self.val_estimates[0]
            ) + \
            (1 - self.prob_to_state) * (
                _lambda ** 0 * self.rewards[1] + \
                _lambda ** 1 * self.rewards[3] + \
                _lambda ** 2 * self.rewards[4] + \
                _lambda ** 3 * self.rewards[5] + \
                _lambda ** 4 * self.rewards[6] + \
                _lambda ** 0 * (1 - _lambda) * self.val_estimates[2] + \
                _lambda ** 1 * (1 - _lambda) * self.val_estimates[3] + \
                _lambda ** 2 * (1 - _lambda) * self.val_estimates[4] + \
                _lambda ** 3 * (1 - _lambda) * self.val_estimates[5] + \
                _lambda ** 4 * (1 - _lambda) * self.val_estimates[6] - \
                self.val_estimates[0]
            )

        return v_s0

    def get_lambda(self, x0=np.linspace(0.1, 1, 7)):
        '''Get lambda value'''
        return fsolve(lambda _: self.get_v_s0(_) - self.get_v_s0(1), x0)

def main():
    '''Main function'''
    homework_problems = [{ # 1
        'prob_to_state': 0.46,
        'val_estimates': [11.7, 13, 0, 24.3, 19.3, -3.7, 0.0],
        'rewards': [0.9, -2.2, 4.2, -1.4, 1, 0, 7.2],
    }, { # 2
        'prob_to_state': 0.0,
        'val_estimates': [0.0, 20.9, 8.9, 0.7, 0, 10, 0.0],
        'rewards': [5.1, 8.3, -2, 0.8, 0, 1.3, 6.2],
    }, { # 3
        'prob_to_state': 0.37,
        'val_estimates': [0.0, 0.4, -3.6, 19.2, 19.2, -4.8, 0.0],
        'rewards': [6.1, 9.6, 2.6, 0, 2.1, 0.2, -1.1],
    }, { # 4
        'prob_to_state': 0.92,
        'val_estimates': [0.0, 13.1, 1.3, 24.5, 19.9, 21.5, 0.0],
        'rewards': [5.3, 0.6, 6.7, -2.2, 8.6, 4.1, -3.2],
    }, { # 5
        'prob_to_state': 0.61,
        'val_estimates': [-3.2, 0, -3.5, 19.9, 24.2, 21.4, 0.0],
        'rewards': [7.5, 9.6, 1.6, 2.9, -1.2, 9.5, -2.3],
    }, { # 6
        'prob_to_state': 0.98,
        'val_estimates': [4.2, 21.7, 21.6, 0, 22.5, 0, 0.0],
        'rewards': [0.8, -2.8, -4.9, 0.3, 0.9, 7.7, 8.9],
    }, { # 7
        'prob_to_state': 0.0,
        'val_estimates': [21.9, -1.1, 0, 23.2, 12.8, 0, 0.0],
        'rewards': [1.0, 4.5, 0, 2.8, 3.5, 2.5, 1.2],
    }, { # 8
        'prob_to_state': 0.09,
        'val_estimates': [19.0, 0, 14, 4.7, 7.9, 13.8, 0.0],
        'rewards': [-0.4, -1.1, 9.7, 7.4, 1.6, 0.5, 3.6],
    }, { # 9
        'prob_to_state': 0.78,
        'val_estimates': [0.0, -1.4, 21, 18.7, 24, 0, 0.0],
        'rewards': [-3.2, -1.5, 5.9, 2, 0, 4, -0.7],
    }, { # 10
        'prob_to_state': 0.87,
        'val_estimates': [13.4, 0.8, -1.1, 19.4, 19.4, 0, 0.0],
        'rewards': [9.7, 2.5, 8.5, 9.8, 0, -2.9, -2.5],
    }]

    for i, problem in enumerate(homework_problems):
        tdl = TD_lambda(problem['prob_to_state'], problem['val_estimates'], problem['rewards'])
        result = tdl.get_lambda()
        print('Problem {} answer: {}'.format(i + 1, min(result)))

if __name__ == "__main__":
    main()

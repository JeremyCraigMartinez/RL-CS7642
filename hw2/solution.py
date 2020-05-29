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

    def get_vs0(self, lambda_):
        '''Get value of initial state in MDP'''
        prob_to_state = self.prob_to_state
        val_estimates = self.val_estimates
        rewards = self.rewards
        # pylint: disable=invalid-name
        vs = dict(zip(['vs0', 'vs1', 'vs2', 'vs3', 'vs4', 'vs5', 'vs6'], list(val_estimates)))

        # pylint: disable=line-too-long
        vs0 = vs['vs0'] + prob_to_state*(rewards[0]+lambda_*rewards[2]+lambda_**2*rewards[4]+lambda_**3*rewards[5]+lambda_**4*rewards[6]+lambda_**4*vs['vs6']+lambda_**3*(1-lambda_)*vs['vs5']+lambda_**2*(1-lambda_)*vs['vs4']+lambda_*(1-lambda_)*vs['vs3']+(1-lambda_)*vs['vs1']-vs['vs0']) +(1-prob_to_state)*(rewards[1]+lambda_*rewards[3]+lambda_**2*rewards[4]+lambda_**3*rewards[5]+lambda_**4*rewards[6]+lambda_**4*vs['vs6']+lambda_**3*(1-lambda_)*vs['vs5']+lambda_**2*(1-lambda_)*vs['vs4']+lambda_*(1-lambda_)*vs['vs3']+(1-lambda_)*vs['vs2']-vs['vs0'])

        return vs0

    def get_lambda(self, x0=np.linspace(0.1, 1, 10)):
        '''Get lambda value'''
        return fsolve(lambda _: self.get_vs0(_) - self.get_vs0(1), x0)

def main():
    '''Main function'''
    print('Run homework problems here')

if __name__ == "__main__":
    main()

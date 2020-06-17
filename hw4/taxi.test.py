'''Unit tests for Agent in frozen_lake'''

import math
import unittest
from taxi import Q

EPSILON_ALPHA_VALUES = ((0.3, 0.2),)
Q_VALUES = {}

class TestMethods(unittest.TestCase):
    '''Unit tests for HW3 solution'''
    # pylint: disable=no-method-argument
    def setUpClass():
        '''Train the Q function before all the tests'''
        # pylint: disable=global-statement
        global Q_VALUES
        for epsilon, alpha in EPSILON_ALPHA_VALUES:
            q_value = Q(epsilon, alpha)
            q_value.train()
            Q_VALUES['ε{}-α{}'.format(epsilon, alpha)] = q_value
            print('Ran Q learning with epsilon: {} and alpha: {}'.format(epsilon, alpha))

    # pylint: disable=no-self-use
    def test_practice_input_1(self):
        '''test_practice_input_1'''
        results = []
        answer = -11.374
        for key, q_value in Q_VALUES.items():
            result = q_value.test(462, 4)
            results.append(result)
            print('For {} - Result {}, expected: {}'.format(key, result, answer))
        assert math.isclose(result, answer, rel_tol=0.001)

    # pylint: disable=no-self-use
    def test_practice_input_2(self):
        '''test_practice_input_2'''
        results = []
        answer = 4.348
        for key, q_value in Q_VALUES.items():
            result = q_value.test(398, 3)
            results.append(result)
            print('For {} - Result {}, expected: {}'.format(key, result, answer))
        assert math.isclose(result, answer, rel_tol=0.001)

    # pylint: disable=no-self-use
    def test_practice_input_3(self):
        '''test_practice_input_3'''
        results = []
        answer = -0.585
        for key, q_value in Q_VALUES.items():
            result = q_value.test(253, 0)
            results.append(result)
            print('For {} - Result {}, expected: {}'.format(key, result, answer))
        assert math.isclose(result, answer, rel_tol=0.001)

    # pylint: disable=no-self-use
    def test_practice_input_4(self):
        '''test_practice_input_4'''
        results = []
        answer = 9.683
        for key, q_value in Q_VALUES.items():
            result = q_value.test(377, 1)
            results.append(result)
            print('For {} - Result {}, expected: {}'.format(key, result, answer))
        assert math.isclose(result, answer, rel_tol=0.001)

    # pylint: disable=no-self-use
    def test_practice_input_5(self):
        '''test_practice_input_5'''
        results = []
        answer = -13.996
        for key, q_value in Q_VALUES.items():
            result = q_value.test(83, 5)
            results.append(result)
            print('For {} - Result {}, expected: {}'.format(key, result, answer))
        assert math.isclose(result, answer, rel_tol=0.001)

if __name__ == '__main__':
    unittest.main()

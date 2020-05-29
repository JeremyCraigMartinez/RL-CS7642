'''HW2 Tests'''

import unittest
from solution import TD_lambda

TESTS = [
    {
        'input': {
            'prob_to_state': 0.81,
            'val_estimates': [0.0, 4.0, 25.7, 0.0, 20.1, 12.2, 0.0],
            'rewards': [7.9, -5.1, 2.5, -7.2, 9.0, 0.0, 1.6],
        },
        'output': 0.623,
    }, {
        'input': {
            'prob_to_state': 0.22,
            'val_estimates': [12.3, -5.2, 0.0, 25.4, 10.6, 9.2, 0.0],
            'rewards': [-2.4, 0.8, 4.0, 2.5, 8.6, -6.4, 6.1],
        },
        'output': 0.52
    }, {
        'input': {
            'prob_to_state': 0.64,
            'val_estimates': [-6.5, 4.9, 7.8, -2.3, 25.5, -10.2, 0.0],
            'rewards': [-2.4, 9.6, -7.8, 0.1, 3.4, -2.1, 7.9],
        },
        'output': 0.208
    }
]

class TestStringMethods(unittest.TestCase):
    '''Unit tests for HW2 solution'''
    def test_practice_value_1(self):
        '''Test first practice problem'''
        _in = TESTS[0]['input']
        _out = TESTS[0]['output']
        tdl = TD_lambda(_in['prob_to_state'], _in['val_estimates'], _in['rewards'])
        result = tdl.get_lambda()
        self.assertEqual(round(result[2], 3), _out)

    def test_practice_value_2(self):
        '''Test second practice problem'''
        _in = TESTS[1]['input']
        _out = TESTS[1]['output']
        tdl = TD_lambda(_in['prob_to_state'], _in['val_estimates'], _in['rewards'])
        result = tdl.get_lambda()
        self.assertEqual(round(result[2], 3), _out)

    def test_practice_value_3(self):
        '''Test third practice problem'''
        _in = TESTS[2]['input']
        _out = TESTS[2]['output']
        tdl = TD_lambda(_in['prob_to_state'], _in['val_estimates'], _in['rewards'])
        result = tdl.get_lambda()
        self.assertEqual(round(result[2], 3), _out)

if __name__ == '__main__':
    unittest.main()

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
    }, { # 1
        'input': {
            'prob_to_state': 0.46,
            'val_estimates': [11.7, 13, 0, 24.3, 19.3, -3.7, 0.0],
            'rewards': [0.9, -2.2, 4.2, -1.4, 1, 0, 7.2],
        },
        'output': 0.18871619875898166,
    }, { # 2
        'input': {
            'prob_to_state': 0.0,
            'val_estimates': [0.0, 20.9, 8.9, 0.7, 0, 10, 0.0],
            'rewards': [5.1, 8.3, -2, 0.8, 0, 1.3, 6.2],
        },
        'output': 0.08125333774816824,
    }, { # 3
        'input': {
            'prob_to_state': 0.37,
            'val_estimates': [0.0, 0.4, -3.6, 19.2, 19.2, -4.8, 0.0],
            'rewards': [6.1, 9.6, 2.6, 0, 2.1, 0.2, -1.1],
        },
        'output': 0.19638090461715896,
    }, { # 4
        'input': {
            'prob_to_state': 0.92,
            'val_estimates': [0.0, 13.1, 1.3, 24.5, 19.9, 21.5, 0.0],
            'rewards': [5.3, 0.6, 6.7, -2.2, 8.6, 4.1, -3.2],
        },
        'output': 0.1746967084235639,
    }, { # 5
        'input': {
            'prob_to_state': 0.61,
            'val_estimates': [-3.2, 0, -3.5, 19.9, 24.2, 21.4, 0.0],
            'rewards': [7.5, 9.6, 1.6, 2.9, -1.2, 9.5, -2.3],
        },
        'output': 0.3915623831120625,
    }, { # 6
        'input': {
            'prob_to_state': 0.98,
            'val_estimates': [4.2, 21.7, 21.6, 0, 22.5, 0, 0.0],
            'rewards': [0.8, -2.8, -4.9, 0.3, 0.9, 7.7, 8.9],
        },
        'output': 0.5327335315636803,
    }, { # 7
        'input': {
            'prob_to_state': 0.0,
            'val_estimates': [21.9, -1.1, 0, 23.2, 12.8, 0, 0.0],
            'rewards': [1.0, 4.5, 0, 2.8, 3.5, 2.5, 1.2],
        },
        'output': 0.4948266305017583,
    }, { # 8
        'input': {
            'prob_to_state': 0.09,
            'val_estimates': [19.0, 0, 14, 4.7, 7.9, 13.8, 0.0],
            'rewards': [-0.4, -1.1, 9.7, 7.4, 1.6, 0.5, 3.6],
        },
        'output': 0.3526026682051179,
    }, { # 9
        'input': {
            'prob_to_state': 0.78,
            'val_estimates': [0.0, -1.4, 21, 18.7, 24, 0, 0.0],
            'rewards': [-3.2, -1.5, 5.9, 2, 0, 4, -0.7],
        },
        'output': 0.23669132125151257,
    }, { # 10
        'input': {
            'prob_to_state': 0.87,
            'val_estimates': [13.4, 0.8, -1.1, 19.4, 19.4, 0, 0.0],
            'rewards': [9.7, 2.5, 8.5, 9.8, 0, -2.9, -2.5],
        },
        'output': 0.09951379112468699,
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
        self.assertEqual(round(min(result), 3), _out)

    def test_practice_value_2(self):
        '''Test second practice problem'''
        _in = TESTS[1]['input']
        _out = TESTS[1]['output']
        tdl = TD_lambda(_in['prob_to_state'], _in['val_estimates'], _in['rewards'])
        result = tdl.get_lambda()
        self.assertEqual(round(min(result), 3), _out)

    def test_practice_value_3(self):
        '''Test third practice problem'''
        _in = TESTS[2]['input']
        _out = TESTS[2]['output']
        tdl = TD_lambda(_in['prob_to_state'], _in['val_estimates'], _in['rewards'])
        result = tdl.get_lambda()
        self.assertEqual(round(min(result), 3), _out)

    # Homework solutions
    def test_hw_value_1(self):
        '''Test first hw problem'''
        _in = TESTS[3]['input']
        _out = TESTS[3]['output']
        tdl = TD_lambda(_in['prob_to_state'], _in['val_estimates'], _in['rewards'])
        result = tdl.get_lambda()
        self.assertEqual(min([r for r in result if r > 0]), _out)

    def test_hw_value_2(self):
        '''Test first hw problem'''
        _in = TESTS[4]['input']
        _out = TESTS[4]['output']
        tdl = TD_lambda(_in['prob_to_state'], _in['val_estimates'], _in['rewards'])
        result = tdl.get_lambda()
        self.assertEqual(min(result), _out)

    def test_hw_value_3(self):
        '''Test first hw problem'''
        _in = TESTS[5]['input']
        _out = TESTS[5]['output']
        tdl = TD_lambda(_in['prob_to_state'], _in['val_estimates'], _in['rewards'])
        result = tdl.get_lambda()
        self.assertEqual(min(result), _out)

    def test_hw_value_4(self):
        '''Test first hw problem'''
        _in = TESTS[6]['input']
        _out = TESTS[6]['output']
        tdl = TD_lambda(_in['prob_to_state'], _in['val_estimates'], _in['rewards'])
        result = tdl.get_lambda()
        self.assertEqual(min(result), _out)

    def test_hw_value_5(self):
        '''Test first hw problem'''
        _in = TESTS[7]['input']
        _out = TESTS[7]['output']
        tdl = TD_lambda(_in['prob_to_state'], _in['val_estimates'], _in['rewards'])
        result = tdl.get_lambda()
        self.assertEqual(min(result), _out)

    def test_hw_value_6(self):
        '''Test first hw problem'''
        _in = TESTS[8]['input']
        _out = TESTS[8]['output']
        tdl = TD_lambda(_in['prob_to_state'], _in['val_estimates'], _in['rewards'])
        result = tdl.get_lambda()
        self.assertEqual(min(result), _out)

    def test_hw_value_7(self):
        '''Test first hw problem'''
        _in = TESTS[9]['input']
        _out = TESTS[9]['output']
        tdl = TD_lambda(_in['prob_to_state'], _in['val_estimates'], _in['rewards'])
        result = tdl.get_lambda()
        self.assertEqual(min(result), _out)

    def test_hw_value_8(self):
        '''Test first hw problem'''
        _in = TESTS[10]['input']
        _out = TESTS[10]['output']
        tdl = TD_lambda(_in['prob_to_state'], _in['val_estimates'], _in['rewards'])
        result = tdl.get_lambda()
        self.assertEqual(min(result), _out)

    def test_hw_value_9(self):
        '''Test first hw problem'''
        _in = TESTS[11]['input']
        _out = TESTS[11]['output']
        tdl = TD_lambda(_in['prob_to_state'], _in['val_estimates'], _in['rewards'])
        result = tdl.get_lambda()
        self.assertEqual(min(result), _out)

    def test_hw_value_10(self):
        '''Test first hw problem'''
        _in = TESTS[12]['input']
        _out = TESTS[12]['output']
        tdl = TD_lambda(_in['prob_to_state'], _in['val_estimates'], _in['rewards'])
        result = tdl.get_lambda()
        self.assertEqual(min(result), _out)

if __name__ == '__main__':
    unittest.main()

'''Unit tests for Agent in frozen_lake'''

import unittest
import numpy as np
from frozen_lake import sarsa
import util

class TestMethods(unittest.TestCase):
    '''Unit tests for HW2 solution'''
    def test_practice_input_1(self):
        '''test_practice_input_1'''
        result = sarsa(
            'SFFG',
            1.0,
            0.24,
            0.09,
            49553,
            202404,
        )
        answer = '<<v<'
        mapped_answer = util.square(answer)
        util.logger('Answer policy:', level='high')
        util.logger(np.array([mapped_answer]).T, level='high')
        self.assertEqual(result, answer)

    def test_practice_input_2(self):
        '''test_practice_input_2'''
        result = sarsa(
            'SFFFHFFFFFFFFFFG',
            1.0,
            0.25,
            0.29,
            14697,
            741684,
        )
        answer = '^vv><>>vvv>v>>><'
        mapped_answer = util.square(answer)
        util.logger('Answer policy:', level='high')
        util.logger(np.array([mapped_answer]).T, level='high')
        self.assertEqual(result, answer)

    def test_practice_input_3(self):
        '''test_practice_input_3'''
        result = sarsa(
            'SFFFFHFFFFFFFFFFFFFFFFFFG',
            0.91,
            0.12,
            0.13,
            42271,
            983459,
        )
        answer = '^>>>><>>>vvv>>vv>>>>v>>^<'
        mapped_answer = util.square(answer)
        util.logger('Answer policy:', level='high')
        util.logger(np.array([mapped_answer]).T, level='high')
        self.assertEqual(result, answer)

if __name__ == '__main__':
    unittest.main()

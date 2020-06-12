'''Unit tests for Agent in frozen_lake'''

import unittest
from frozen_lake import Agent

class TestMethods(unittest.TestCase):
    '''Unit tests for HW2 solution'''
    def test_practice_input_1(self):
        '''test_practice_input_1'''
        agent = Agent(
            amap='SFFG',
            gamma=1.0,
            alpha=0.24,
            epsilon=0.09,
            num_episodes=49553,
            seed=202404,
        )
        path = agent.train()
        self.assertEqual(path, '<<v<')

    def test_practice_input_2(self):
        '''test_practice_input_2'''
        agent = Agent(
            amap='SFFFHFFFFFFFFFFG',
            gamma=1.0,
            alpha=0.25,
            epsilon=0.29,
            num_episodes=14697,
            seed=741684,
        )
        path = agent.train()
        self.assertEqual(path, '^vv><>>vvv>v>>><')

    def test_practice_input_3(self):
        '''test_practice_input_3'''
        agent = Agent(
            amap='SFFFFHFFFFFFFFFFFFFFFFFFG',
            gamma=0.91,
            alpha=0.12,
            epsilon=0.13,
            num_episodes=42271,
            seed=983459,
        )
        path = agent.train()
        self.assertEqual(path, '^>>>><>>>vvv>>vv>>>>v>>^<')

if __name__ == '__main__':
    unittest.main()

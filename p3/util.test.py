'''Unit tests for Agent in frozen_lake'''

import unittest
from util import SoccerEnv

class TestMethods(unittest.TestCase):
    '''Unit Soccer environment'''
    def test_soccer_env_scenario_1(self):
        '''test_soccer_env'''
        env = SoccerEnv()
        (pos_a, pos_b, possession), rewards, done = env.simulate_action(2, 2)
        self.assertEqual(pos_a.tolist(), [1, 2])
        self.assertEqual(pos_b.tolist(), [1, 1])
        self.assertEqual(possession, 'B')
        self.assertEqual(rewards, {'B': 0, 'A': 0})
        self.assertEqual(done, False)

        (pos_a, pos_b, possession), rewards, done = env.simulate_action(1, 4)
        self.assertEqual(pos_a.tolist(), [1, 3])
        self.assertEqual(pos_b.tolist(), [1, 1])
        self.assertEqual(possession, 'B')
        self.assertEqual(rewards, {'B': 0, 'A': 0})
        self.assertEqual(done, False)

        (pos_a, pos_b, possession), rewards, done = env.simulate_action(4, 3)
        self.assertEqual(pos_a.tolist(), [1, 3])
        self.assertEqual(pos_b.tolist(), [1, 0])
        self.assertEqual(possession, 'B')
        self.assertEqual(rewards, {'B': -100, 'A': 100})
        self.assertEqual(done, True)

        # Throws exception
        exception_raised = False
        try:
            (pos_a, pos_b, possession), rewards, done = env.simulate_action(1, 4)
        # pylint: disable=broad-except
        except Exception:
            exception_raised = True
        self.assertEqual(exception_raised, True)

    def test_soccer_env_scenario_2(self):
        '''test_soccer_env'''
        env = SoccerEnv()
        (pos_a, pos_b, possession), rewards, done = env.simulate_action(3, 1)
        try:
            self.assertEqual(pos_a.tolist(), [0, 1])
            self.assertEqual(pos_b.tolist(), [0, 1])
            self.assertEqual(possession, 'A')
        except AssertionError:
            self.assertEqual(pos_a.tolist(), [0, 2])
            self.assertEqual(pos_b.tolist(), [0, 2])
            self.assertEqual(possession, 'B')
        self.assertEqual(rewards, {'B': 0, 'A': 0})
        self.assertEqual(done, False)

    def test_soccer_env_scenario_3(self):
        '''test_soccer_env'''
        env = SoccerEnv()
        (pos_a, pos_b, possession), rewards, done = env.simulate_action(4, 2)
        self.assertEqual(pos_a.tolist(), [0, 2])
        self.assertEqual(pos_b.tolist(), [1, 1])
        self.assertEqual(possession, 'B')
        self.assertEqual(rewards, {'B': 0, 'A': 0})
        self.assertEqual(done, False)

        (pos_a, pos_b, possession), rewards, done = env.simulate_action(4, 1)
        self.assertEqual(pos_a.tolist(), [0, 2])
        self.assertEqual(pos_b.tolist(), [1, 2])
        self.assertEqual(possession, 'B')
        self.assertEqual(rewards, {'B': 0, 'A': 0})
        self.assertEqual(done, False)

        (pos_a, pos_b, possession), rewards, done = env.simulate_action(4, 1)
        self.assertEqual(pos_a.tolist(), [0, 2])
        self.assertEqual(pos_b.tolist(), [1, 3])
        self.assertEqual(possession, 'B')
        self.assertEqual(rewards, {'B': 100, 'A': -100})
        self.assertEqual(done, True)

if __name__ == '__main__':
    unittest.main()

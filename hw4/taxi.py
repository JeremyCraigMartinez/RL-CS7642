'''HW3 solution'''

from sys import argv
import json
import numpy as np
import gym
import util

# pylint: disable=invalid-name
class Q():
    '''Q learning class'''
    def __init__(self, epsilon, alpha):
        env = gym.make('Taxi-v3')
        seed = 202404
        env.seed(seed)
        np.random.seed(seed)

        self.q_value = np.zeros([env.observation_space.n, env.action_space.n])
        self.env = env
        self.gamma = 0.9
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_episodes = 250000

    def train(self):
        '''Q learning algorithm'''

        for i in range(self.num_episodes):
            state = self.env.reset()
            if i % 10000 == 0 and i > 0:
                util.logger('Episode {}...'.format(i))

            done = False
            while not done:
                if '--render' in argv:
                    self.env.render()
                # pylint: disable=line-too-long
                action = util.select_epsilon_greedy_action(state, self.q_value, self.epsilon, self.env)
                next_state, reward, done, _ = self.env.step(action)
                next_state_max_action = util.select_greedy_action(next_state, self.q_value)
                # pylint: disable=line-too-long
                update = self.alpha * (reward + self.gamma * self.q_value[next_state, next_state_max_action] - self.q_value[state, action])
                self.q_value[state, action] += update
                state = next_state

        self.env.close()

    def test(self, state, action):
        '''Test trained Q function'''
        return self.q_value[state, action]

def main():
    '''Main function to test when invoked directly'''
    q = Q(0.3, 0.2)
    q.train()

    with open('homework-problems.json') as f:
        hw_problems = json.load(f)
        i = 0
        for problem in hw_problems:
            i = i + 1
            result = q.test(problem['state'], problem['action'])
            print('The answer to problem {}: {}'.format(i, result))

if __name__ == "__main__":
    main()

'''Solution to HW3'''

import json
import numpy as np
import gym
import util

# pylint: disable=too-few-public-methods,too-many-instance-attributes
class Agent():
    '''Agent to train on OpenAI frozen lake'''
    def __init__(self, **kwargs):
        self.amap = util.square_amap(kwargs['amap'])
        self.gamma = kwargs['gamma']
        self.alpha = kwargs['alpha']
        self.epsilon = kwargs['epsilon']
        self.num_episodes = kwargs['num_episodes']
        self.seed = kwargs['seed']
        self.env = gym.make('FrozenLake-v0', desc=self.amap).unwrapped
        self.actions = {
            'Left': 0,
            'Down': 1,
            'Right': 2,
            'Up': 3
        }

        np.random.seed(kwargs['seed'])
        self.env.seed(kwargs['seed'])

    def init_Q(self):
        '''initialize Q function'''
        # pylint: disable=invalid-name
        Q = []
        for row in self.amap:
            Q.append([])
            for _ in row:
                Q[-1].append(int(np.random.random() * 4))
        return Q


    def train(self):
        '''Train Agent to navigate the frozen lake'''
        env = self.env
        env.reset()
        # pylint: disable=invalid-name
        Q = self.init_Q()

        for _ in range(self.num_episodes):
            observation, reward, done, info = env.step(env.action_space.sample())

            if done:
                util.logger('Episode finished after {} timesteps'.format(_ + 1))

        env.close()

        return ''

if __name__ == '__main__':
    with open('homework-problems.json', 'w') as homework_problems:
        # pylint: disable=invalid-name
        problems = json.load(homework_problems)
        for problem in problems:
            agent = Agent(
                amap=problem['amap'],
                gamma=problem['gamma'],
                alpha=problem['alpha'],
                epsilon=problem['epsilon'],
                num_episodes=problem['num_episodes'],
                seed=problem['seed'],
            )
            agent.train()

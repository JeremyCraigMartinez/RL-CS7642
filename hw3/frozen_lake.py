'''HW3 solution'''

from math import sqrt
import json
import numpy as np
import util
import gym

def sarsa(amap, gamma, alpha, epsilon, num_episodes, seed):
    '''SARSA algorithm'''
    env = gym.make('FrozenLake-v0', desc=util.square(amap)).unwrapped

    env.seed(seed)
    np.random.seed(seed)

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    for _ in range(num_episodes):
        state = env.reset()
        action = util.select_action(state, Q, epsilon, env)

        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = util.select_action(next_state, Q, epsilon, env)
            update = reward + gamma * Q[next_state, next_action] - Q[state, action]
            Q[state, action] = Q[state, action] + alpha * update
            state = next_state
            action = next_action

    env.close()
    Q_star = np.argmax(Q, axis=1)
    mapping = ['<', 'v', '>', '^']
    return ''.join([mapping[action] for action in Q_star])

if __name__ == "__main__":
    with open('homework-problems.json') as f:
        hw_problems = json.load(f)
        i = 0
        for problem in hw_problems:
            i = i + 1
            result = sarsa(
                problem['amap'],
                problem['gamma'],
                problem['alpha'],
                problem['epsilon'],
                problem['num_episodes'],
                problem['seed'],
            )
            print('The answer to problem {}: {}'.format(i, result))

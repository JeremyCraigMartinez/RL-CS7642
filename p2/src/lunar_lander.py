'''Project 2 Deep Q-Network Lunar Lander'''

import json
from sys import argv
from timeit import default_timer as timer
import random
from collections import deque
import numpy as np
import gym
import tensorflow as tf
import util

tf.compat.v1.disable_eager_execution()
TARGET_SCORE = 200
OPEN_AI_ENV = 'LunarLander-v2'

# pylint: disable=too-many-instance-attributes
class DQN:
    '''Deep Q-Networks'''
    def __init__(self, gamma=0.99, epsilon_decay=0.005):
        seed = 983827
        self.env = gym.make(OPEN_AI_ENV)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.env.seed(seed)
        np.random.seed(seed)
        self.replay_memory = deque(maxlen=int(1000000))
        self.gamma = gamma
        self.learning_rate = 0.0005
        self.alpha = 0.003
        self.action_value_function = self.get_model()
        self.target_action_value_function = self.get_model()
        self.update_frequency = 4
        self.batch_size = 64
        self.steps = 0
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay

    def get_model(self):
        '''Get SGD model'''
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim=self.state_space, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_space))
        # pylint: disable=line-too-long
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def select_epsilon_greedy_action(self, state):
        '''Select greedy action with epsilon probability'''
        # pylint: disable=no-member
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.action_value_function.predict(state)[0])

    def sample_minibatch(self):
        '''Sample mini batch'''
        if self.steps % self.update_frequency != 0:
            return

        states = []
        targets = []
        samples = random.sample(self.replay_memory, self.batch_size)

        for state, action, reward, next_state, done in samples:
            if done:
                q_hat_update = reward
            else:
                q_hat_update = reward + self.gamma * \
                    np.amax(self.target_action_value_function.predict(next_state)[0])

            q_target = self.action_value_function.predict(state)
            q_target[0][action] = q_hat_update
            states.append(state[0])
            targets.append(q_target[0])

        self.action_value_function.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        # update weights
        weights = self.action_value_function.get_weights()
        target_weights = self.target_action_value_function.get_weights()
        import pdb; pdb.set_trace()
        for i, _ in enumerate(target_weights):
            target_weights[i] = self.alpha * weights[i] + (1 - self.alpha) * target_weights[i]
        self.target_action_value_function.set_weights(target_weights)

# pylint: disable=too-many-locals
def train(dqn, num_episodes=2000, steps=500, filename=None):
    '''Train DQN'''
    scores = []
    episode_elapsed_time = deque(maxlen=100)
    epsilons = []
    agg_steps = []

    for episode in range(num_episodes):
        overall_reward = 0
        state = dqn.env.reset().reshape(1, dqn.state_space)
        epsilons.append(dqn.epsilon)
        start = timer()

        for i in range(steps):
            dqn.steps += 1
            action = dqn.select_epsilon_greedy_action(state)
            next_state, reward, done, _ = dqn.env.step(action)
            overall_reward += reward
            next_state = next_state.reshape(1, dqn.state_space)
            dqn.replay_memory.append([state, action, reward, next_state, done])

            if len(dqn.replay_memory) > dqn.batch_size:
                dqn.sample_minibatch()
            state = next_state

            if done:
                break

        agg_steps.append(i)
        scores.append(overall_reward)
        episode_elapsed_time.append(timer() - start)

        if dqn.epsilon > 0.01:
            dqn.epsilon *= (1 - dqn.epsilon_decay)

        # pylint: disable=line-too-long
        info = '\rEpisode {}\t Mean Score: {:.3f} \t Epsilon: {:.3f} \t Avg time: {:.3f} \t Avg steps: {:.3f}'.format(episode, np.mean(scores[-100:]), dqn.epsilon, np.mean(episode_elapsed_time), np.mean(agg_steps))
        if episode % 25 == 0:
            print(info)
        else:
            print(info, end='')

        if np.mean(scores[-100:]) >= TARGET_SCORE:
            if filename:
                dqn.action_value_function.save(filename)
            break

    dqn.env.close()
    return (scores, epsilons)

# pylint: disable=too-many-locals
def test(filename='DQN.model'):
    '''Test DQN'''
    model = tf.keras.models.load_model(filename)
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True, to_file='DQN.model.png')
    episode_elapsed_time = deque(maxlen=100)
    env = gym.make(OPEN_AI_ENV)
    scores = []
    state_space = env.observation_space.shape[0]

    for episode in range(100):
        start = timer()
        overall_reward = 0
        state = env.reset().reshape(1, state_space)
        done = False
        while not done:
            env.render()
            # same line as in epsilon greedy action selection
            action = np.argmax(model.predict(state)[0])
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1, state_space)
            overall_reward += reward
            state = next_state

        episode_elapsed_time.append(timer() - start)
        scores.append(overall_reward)
        # pylint: disable=line-too-long
        info = '\rEpisode {}\t Mean Score: {:.3f} \t Avg episode time: {:.3f}'.format(episode, np.mean(scores), np.mean(episode_elapsed_time))
        print(info, end='')

    env.close()
    return scores

def main():
    '''Main driver for script'''
    agent = DQN()
    filename = 'DQN.model'
    if '--model-file' in argv:
        index = argv.index('--model-file')
        filename = argv[index + 1]
    if not '--skip-training' in argv:
        train_scores, epsilons = train(agent, filename=filename)
        with open('agent-scores.json', 'w') as outfile:
            json.dump(train_scores, outfile)
        util.plot_two_separate([train_scores, epsilons], 'Episodes', ['Training Scores', 'Epsilon Values'], 'training')
        if '--train-only' in argv:
            return

    test_scores = test(filename=filename)
    util.plot_values(test_scores, 'Episodes', 'Scores', 'testing')

def test_gamma_hyperparameters():
    index = argv.index('--gamma') + 1
    gamma = float(argv[index])
    agent = DQN(gamma=gamma)
    train_scores, _ = train(agent, num_episodes=3)
    with open('agent-train-hyperparameter-epsilon-decay-{}.json'.format(gamma), 'w') as outfile:
        json.dump(train_scores, outfile)

def test_epsilon_decay_hyperparameters():
    index = argv.index('--epsilon-decay') + 1
    epsilon_decay = float(argv[index])
    agent = DQN(epsilon_decay=epsilon_decay)
    train_scores, _ = train(agent)
    with open('agent-train-hyperparameter-epsilon-decay-{}.json'.format(epsilon_decay), 'w') as outfile:
        json.dump(train_scores, outfile)

if __name__ == '__main__':
    if '--hyperparameter-testing' in argv:
        if '--gamma' in argv:
            test_gamma_hyperparameters()
        elif '--epsilon-decay' in argv:
            test_epsilon_decay_hyperparameters()
        elif '--layers' in argv:
            test_layers_hyperparameters()
    else:
        main()

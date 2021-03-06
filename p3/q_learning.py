'''Q-learning'''

from sys import argv
import pickle
import numpy as np
from util import SoccerEnv, logger, plot_fig, translate_rewards, translate_state, map_action

def select_epsilon_greedy_action(Q, state, epsilon):
    '''Select epsilon greedy action'''
    if np.random.random() < epsilon:
        return np.random.randint(5)
    A, B, possession = state
    max_action = np.where(Q[A][B][possession] == max(Q[A][B][possession]))[0]
    return np.random.choice(max_action, 1)[0]

# pylint: disable=too-many-locals
def q_learning(steps=1000000):
    '''Q-learning'''

    # define hyperparameters
    gamma = .9
    epsilon = 1
    epsilon_min = .001
    epsilon_decay = .000005
    alpha = 1
    alpha_min = .001
    alpha_decay = .000005
    q_value_diff = []

    # length two for number of players
    q_values = [
        np.zeros((8, 8, 2, 5)),
        np.zeros((8, 8, 2, 5)),
    ]

    for _ in range(steps):
        if _ > 0 and _ % 1000 == 0:
            if _ % 10000 == 0:
                print('\rStep {}\tEpsilon: {:.3f}\tLast error: {}\tError mean: {:.3f}\tPercentage: {:.2f}%'.format(_, epsilon, q_value_diff[-1], np.mean(q_value_diff), _ * 100 / steps))
            else:
                print('\rStep {}\tEpsilon: {:.3f}\tLast error: {}\tError mean: {:.3f}\tPercentage: {:.2f}%'.format(_, epsilon, q_value_diff[-1], np.mean(q_value_diff), _ * 100 / steps), end='')
        env = SoccerEnv()
        done = False
        while not done:
            state = translate_state(env.get_state())
            actions = [select_epsilon_greedy_action(q_value, state, epsilon) for q_value in q_values]
            next_state, _rewards, done = env.simulate_action(*actions)
            rewards = translate_rewards(_rewards)
            if '--render' in argv:
                env.render(actions, done)
            a_pos, b_pos, ball_poss = state
            next_a_pos, next_b_pos, next_ball_poss = translate_state(next_state)
            before = q_values[0][2][1][1][3]
            if done:
                for i, _ in enumerate(q_values):
                    action = actions[i]
                    diff = alpha * (rewards[i] - q_values[i][a_pos][b_pos][ball_poss][action])
                    q_values[i][a_pos][b_pos][ball_poss][action] += diff
            else:
                for i, _ in enumerate(q_values):
                    action = actions[i]
                    diff = alpha * (rewards[i] + gamma * \
                        max(q_values[i][next_a_pos][next_b_pos][next_ball_poss]) - q_values[i][a_pos][b_pos][ball_poss][action])
                    q_values[i][a_pos][b_pos][ball_poss][action] += diff

            after = q_values[0][2][1][1][3]
            if epsilon > epsilon_min:
                epsilon *= (1 - epsilon_decay)
            if alpha > alpha_min:
                alpha *= (1 - alpha_decay)

            state = translate_state(next_state)
            q_value_diff.append(abs(after - before))
            if q_value_diff[-1] > 0:
                logger('Player positions: {}\t Error: {:.3f}\t State: {}; Actions: {};\t Next State: {}; Rewards: {};\t Terminal: {}'.format((a_pos, b_pos), q_value_diff[-1], state, [map_action(a) for a in actions], translate_state(next_state), rewards, done))

    return q_value_diff

if __name__ == '__main__':
    if '--plot-only' in argv:
        list_of_error_diffs = pickle.load(open('bin/q-learner.p', 'rb'))
        plot_fig(np.array(list_of_error_diffs)[np.where(np.array(list_of_error_diffs) > 0)], 'Q-learner')
    else:
        list_of_error_diffs = q_learning()
        pickle.dump(list_of_error_diffs, open('bin/q-learner.p', 'wb'))
        plot_fig(np.array(list_of_error_diffs)[np.where(np.array(list_of_error_diffs) > 0)], 'Q-learner')

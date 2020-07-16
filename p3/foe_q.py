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
    min_action = np.where(Q[A][B][possession] == np.min(Q[A][B][possession]))[1]
    return np.random.choice(min_action, 1)[0]

# pylint: disable=too-many-locals
def foe_q(steps=1000000):
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

    # actions for both players in Q values
    q_values = [
        np.zeros((8, 8, 2, 5, 5)),
        np.zeros((8, 8, 2, 5, 5)),
    ]

    for _ in range(steps):
        if _ % 1000 == 0:
            print('\rPercentage: {:.2f}%'.format(_ * 100 / steps), end='')
        env = SoccerEnv()
        done = False
        while not done:
            state = translate_state(env.get_state())
            actions = [select_epsilon_greedy_action(q_values[1], state, epsilon), select_epsilon_greedy_action(q_values[0], state, epsilon)]
            next_state, _rewards, done = env.simulate_action(*actions)
            rewards = translate_rewards(_rewards)
            logger('State: {}; Actions: {};\t Next State: {}; Rewards: {};\t Terminal: {}'.format(state, [map_action(a) for a in actions], translate_state(next_state), rewards, done))

            a_pos, b_pos, ball_poss = state
            next_a_pos, next_b_pos, next_ball_poss = translate_state(next_state)
            before = q_values[0][a_pos][b_pos][ball_poss][actions[0]][4]
            if done:
                # Player A
                diff = alpha * (rewards[0] - q_values[0][a_pos][b_pos][ball_poss][actions[0]][actions[1]])
                q_values[0][a_pos][b_pos][ball_poss][actions[0]][actions[1]] += diff
                # Player B
                diff = alpha * (rewards[1] - q_values[1][a_pos][b_pos][ball_poss][actions[1]][actions[0]])
                q_values[1][a_pos][b_pos][ball_poss][actions[1]][actions[0]] += diff
            else:
                # Player A
                diff = alpha * (rewards[0] + gamma * \
                    np.max(q_values[0][next_a_pos][next_b_pos][next_ball_poss]) - q_values[0][a_pos][b_pos][ball_poss][actions[0]][actions[1]])
                q_values[0][a_pos][b_pos][ball_poss][actions[0]][actions[1]] += diff
                # Player B
                diff = alpha * (rewards[1] + gamma * \
                    np.max(q_values[1][next_a_pos][next_b_pos][next_ball_poss]) - q_values[1][a_pos][b_pos][ball_poss][actions[1]][actions[0]])
                q_values[1][a_pos][b_pos][ball_poss][actions[1]][actions[0]] += diff
            after = q_values[0][a_pos][b_pos][ball_poss][actions[0]][4]

            if epsilon > epsilon_min:
                epsilon *= (1 - epsilon_decay)
            if alpha > alpha_min:
                alpha *= (1 - alpha_decay)

            state = translate_state(next_state)

        q_value_diff.append(abs(after - before))

    return q_value_diff

if __name__ == '__main__':
    np.random.seed(1827343)
    if '--plot-only' in argv:
        list_of_error_diffs = pickle.load(open('bin/foe-q.p', 'rb'))
        plot_fig(list_of_error_diffs, 'Foe-Q')
    else:
        list_of_error_diffs = foe_q()
        pickle.dump(list_of_error_diffs, open('bin/foe-q.p', 'wb'))
        plot_fig(list_of_error_diffs, 'Foe-Q')

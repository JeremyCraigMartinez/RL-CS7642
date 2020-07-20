'''Q-learning'''

from sys import argv
import pickle
import numpy as np
from scipy.linalg import block_diag
from cvxopt import matrix, solvers
from util import SoccerEnv, logger, plot_fig, translate_rewards, translate_state, map_action
solvers.options['show_progress'] = False

def ce(Q1, Q2, state):
    A, B, possession = state
    row_index = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23])
    col_index = np.array([0, 5, 10, 15, 20, 1, 6, 11, 16, 21, 2, 7, 12, 17, 22, 3, 8, 13, 18, 23, 4, 9, 14, 19, 24])

    s = block_diag(
        Q1[A][B][possession] - Q1[A][B][possession][0, :],
        Q1[A][B][possession] - Q1[A][B][possession][1, :],
        Q1[A][B][possession] - Q1[A][B][possession][2, :],
        Q1[A][B][possession] - Q1[A][B][possession][3, :],
        Q1[A][B][possession] - Q1[A][B][possession][4, :]
    )
    G1 = s[row_index, :]

    s = block_diag(
        Q2[A][B][possession] - Q2[A][B][possession][0, :],
        Q2[A][B][possession] - Q2[A][B][possession][1, :],
        Q2[A][B][possession] - Q2[A][B][possession][2, :],
        Q2[A][B][possession] - Q2[A][B][possession][3, :],
        Q2[A][B][possession] - Q2[A][B][possession][4, :]
    )
    G2 = s[row_index, :][:, col_index]

    c = matrix((Q1[A][B][possession] + Q2[A][B][possession].T).reshape(25))
    G = matrix(
        np.append(
            np.append(
                G1,
                G2,
                axis=0
            ),
            -np.eye(25),
            axis=0
        )
    )
    h = matrix(np.zeros(65, dtype=float))
    A = matrix(np.ones((1, 25), dtype=float))
    b = matrix(1.)
    try:
        lp = solvers.lp(c=c, G=G, h=h, A=A, b=b)
        return np.abs(np.array(lp['x']).reshape((5, 5))) / sum(np.abs(lp['x']))
    except Exception as e:
        print('Exception caught from solver: ', e)

    return (np.ones(25) * 0.04) / sum(np.ones(25) * 0.04)

def select_epsilon_greedy_action(Q1, Q2, state, epsilon):
    '''Select epsilon greedy action'''
    if np.random.random() < epsilon:
        return [np.random.randint(5), np.random.randint(5)]
    prob = ce(Q1, Q2, state).reshape(25)
    i = np.random.choice(np.arange(25), 1, p=prob)[0]
    return i // 5, i % 5

# pylint: disable=too-many-locals
def uce_q(steps=1000000):
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
        np.ones((8, 8, 2, 5, 5)) * 1.,
        np.ones((8, 8, 2, 5, 5)) * 1.,
    ]

    for _ in range(steps):
        if _ > 0 and _ % 1000 == 0:
            if _ % 10000 == 0:
                print('\rStep {}\tEpsilon: {:.3f}\tLast error: {}\tError mean: {:.3f}\tPercentage: {:.2f}%\t Alpha: {:.3f}'.format(_, epsilon, q_value_diff[-1], np.mean(q_value_diff[:-100]), _ * 100 / steps, alpha))
            else:
                print('\rStep {}\tEpsilon: {:.3f}\tLast error: {}\tError mean: {:.3f}\tPercentage: {:.2f}%\t Alpha: {:.3f}'.format(_, epsilon, q_value_diff[-1], np.mean(q_value_diff[:-100]), _ * 100 / steps, alpha), end='')
        env = SoccerEnv()
        done = False
        while not done:
            state = translate_state(env.get_state())
            actions = select_epsilon_greedy_action(q_values[0], q_values[1], state, epsilon)
            next_state, _rewards, done = env.simulate_action(*actions)
            rewards = translate_rewards(_rewards)
            a_pos, b_pos, ball_poss = state
            next_a_pos, next_b_pos, next_ball_poss = translate_state(next_state)
            before = q_values[0][2][1][1][3][4] # Per section 5 of Greenwald-Hall-2003

            if done:
                # Player A
                diff = alpha * (rewards[0] - q_values[0][a_pos][b_pos][ball_poss][actions[0]][actions[1]])
                q_values[0][a_pos][b_pos][ball_poss][actions[0]][actions[1]] += diff
                # Player B
                diff = alpha * (rewards[1] - q_values[1][a_pos][b_pos][ball_poss][actions[1]][actions[0]])
                q_values[1][a_pos][b_pos][ball_poss][actions[1]][actions[0]] += diff
                break
            else:
                # Player A
                diff = alpha * (rewards[0] + gamma * \
                    np.max(q_values[0][next_a_pos][next_b_pos][next_ball_poss]) - q_values[0][a_pos][b_pos][ball_poss][actions[0]][actions[1]])
                q_values[0][a_pos][b_pos][ball_poss][actions[0]][actions[1]] = (1 - alpha) * q_values[0][a_pos][b_pos][ball_poss][actions[0]][actions[1]] + diff
                # Player B
                diff = alpha * (rewards[1] + gamma * \
                    np.max(q_values[1][next_a_pos][next_b_pos][next_ball_poss]) - q_values[1][a_pos][b_pos][ball_poss][actions[1]][actions[0]])
                q_values[1][a_pos][b_pos][ball_poss][actions[1]][actions[0]] = (1 - alpha) * q_values[1][a_pos][b_pos][ball_poss][actions[1]][actions[0]] + diff

                if epsilon > epsilon_min:
                    epsilon *= (1 - epsilon_decay)
                if alpha > alpha_min:
                    alpha *= (1 - alpha_decay)

            after = q_values[0][2][1][1][3][4]

            state = translate_state(next_state)
            q_value_diff.append(abs(after - before))
            if q_value_diff[-1] > 0:
                logger('Player positions: {}\t Error: {:.3f}\t State: {}; Actions: {};\t Next State: {}; Rewards: {};\t Terminal: {}'.format((a_pos, b_pos), q_value_diff[-1], state, [map_action(a) for a in actions], translate_state(next_state), rewards, done))

    return np.array(q_value_diff)[np.where(np.array(q_value_diff) > 0)]

if __name__ == '__main__':
    if '--plot-only' in argv:
        list_of_error_diffs = pickle.load(open('bin/ce-q.p', 'rb'))
        plot_fig(list_of_error_diffs, 'Correlated-Q')
    else:
        list_of_error_diffs = uce_q()
        pickle.dump(list_of_error_diffs, open('bin/ce-q.p', 'wb'))
        plot_fig(list_of_error_diffs, 'Correlated-Q')

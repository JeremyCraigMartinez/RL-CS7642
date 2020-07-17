'''Reused utility classes/functions for Project 3'''

from sys import argv
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 100000000000000000

def logger(_str, level='low'):
    '''logger enabled with --verbose flag'''
    if '--verbose' in argv or level == 'high':
        print(_str)

def plot_fig(errors, title):
    '''Plot convergence'''
    plt.plot(errors, 'k', linestyle='-', linewidth=0.6)

    plt.title(title)
    plt.xlabel('Simulation Iteartion')
    plt.ylabel('Q-value Difference')
    plt.ylim(0., .5)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    plt.margins(x=0, y=0)

    fig = plt.gcf()
    plt.draw()
    fig.savefig('figs/{}.png'.format(title), dpi=fig.dpi)

def translate_state(state):
    '''translate state for Q values'''
    A, B, _possession = state
    A_row, A_col = A
    B_row, B_col = B
    A_position = A_row * 4 + A_col
    B_position = B_row * 4 + B_col
    possession = 1 if _possession == 'B' else 0
    return A_position, B_position, possession

def translate_rewards(rewards):
    '''translate rewards from obj to array'''
    return [rewards['A'], rewards['B']]

def map_action(action):
    '''Get human readable action from number'''
    if action == 0:
        return 'up'
    if action == 1:
        return 'right'
    if action == 2:
        return 'down'
    if action == 3:
        return 'left'
    return 'nada'

def apply_action_to_player(player_position, action):
    '''The physics of the world, pertaining to the action'''
    action_diff = (
        np.array([-1, 0]), # Up
        np.array([0, 1]),  # Right
        np.array([1, 0]),  # Down
        np.array([0, -1]), # Left
        np.array([0, 0]),  # Nothing
    )

    # pylint: disable=too-many-boolean-expressions
    if (player_position[0] == 0 and action == 0) or \
        (player_position[1] == 0 and action == 3) or \
        (player_position[0] == 1 and action == 2) or \
        (player_position[1] == 3 and action == 1):
        return player_position

    return player_position + action_diff[action]

class SoccerEnv:
    '''Soccer environment to run learner on'''
    def __init__(self):
        self.player_positions = {
            'A': np.array([0, 2]), # Player A
            'B': np.array([0, 1]), # Player B
        }
        self.possession = 'B'
        self.terminal_states_for_a = np.array([np.array([0, 0]), np.array([1, 0])])
        self.terminal_states_for_b = np.array([np.array([0, 3]), np.array([1, 3])])
        self.done = False

    def get_state(self):
        '''Get state'''
        return (self.player_positions['A'], self.player_positions['B'], self.possession)

    def simulate_action(self, action_a, action_b):
        '''simulate action A in state S, and return state S', reward R, and terminal boolean'''
        if self.done:
            raise Exception('Environment has terminated')

        rewards = {'A': 0, 'B': 0}
        actions = {'A': action_a, 'B': action_b}
        first_player = 'A' if np.random.choice([0, 1], 1)[0] == 0 else 'B'
        second_player = 'A' if first_player == 'B' else 'B'

        if action_a < 0 or action_a > 4 or action_b < 0 or action_b > 4:
            return self.get_state(), rewards, self.done

        # Player A moves first
        new_state = {}
        new_state[first_player] = apply_action_to_player(self.player_positions[first_player], actions[first_player])

        # If they collide, only the first play moves
        if (new_state[first_player] == self.player_positions[second_player]).all():
            #  If the player with the ball moves second, then the ball changes possession
            if self.possession == second_player:
                self.possession = first_player
            self.player_positions[first_player] = new_state[first_player]
            return self.get_state(), rewards, self.done
        new_state[second_player] = apply_action_to_player(self.player_positions[second_player], actions[second_player])

        # Update positions for players in env
        self.player_positions[first_player] = new_state[first_player]
        self.player_positions[second_player] = new_state[second_player]

        if np.any([(_ == self.player_positions['A']).all() for _ in self.terminal_states_for_a]) and self.possession == 'A':
            rewards = {'A': 100, 'B': -100}
            self.done = True
        elif np.any([(_ == self.player_positions['A']).all() for _ in self.terminal_states_for_b]) and self.possession == 'A':
            rewards = {'A': -100, 'B': 100}
            self.done = True
        elif np.any([(_ == self.player_positions['B']).all() for _ in self.terminal_states_for_b]) and self.possession == 'B':
            rewards = {'A': -100, 'B': 100}
            self.done = True
        elif np.any([(_ == self.player_positions['B']).all() for _ in self.terminal_states_for_a]) and self.possession == 'B':
            rewards = {'A': 100, 'B': -100}
            self.done = True

        return self.get_state(), rewards, self.done

    def render(self, actions, done):
        '''Render out environment in console'''
        def get_cell_print_out(row, col):
            # import pdb; pdb.set_trace()
            is_A = (self.player_positions['A'] == np.array([row, col])).all()
            is_B = (self.player_positions['B'] == np.array([row, col])).all()
            if is_A and is_B:
                if self.possession == 'A':
                    return 'AB'
                return 'BA'
            if is_A:
                if self.possession == 'A':
                    return 'A*'
                return 'A '
            if is_B:
                if self.possession == 'B':
                    return 'B*'
                return 'B '
            return '  '

        print('''Player A moved {}; Player B moved {}:
+-----------+
|{}|{}|{}|{}|
+--+--+--+--+
|{}|{}|{}|{}|
+-----------+
{}'''.format(
    map_action(actions[0]), map_action(actions[1]),
    get_cell_print_out(0, 0),
    get_cell_print_out(0, 1),
    get_cell_print_out(0, 2),
    get_cell_print_out(0, 3),
    get_cell_print_out(1, 0),
    get_cell_print_out(1, 1),
    get_cell_print_out(1, 2),
    get_cell_print_out(1, 3),
    '' if not done else 'Game ends'
))

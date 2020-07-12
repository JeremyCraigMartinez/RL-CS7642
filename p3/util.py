'''Reused utility classes/functions for Project 3'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000000

def plot_fig(errors, title):
    '''Plot convergence'''
    plt.plot(errors, linestyle='-', linewidth=0.6)

    plt.title(title)
    plt.xlabel('Simulation Iteartion')
    plt.ylabel('Q-value Difference')
    plt.ylim(0., .5)

    plt.show()

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

        action_diff = (
            np.array([-1, 0]), # Up
            np.array([0, 1]),  # Right
            np.array([1, 0]),  # Down
            np.array([0, -1]), # Left
            np.array([0, 0]),  # Nothing
        )
        rewards = {'A': 0, 'B': 0}
        actions = {'A': action_a, 'B': action_b}
        first_player = 'A' if np.random.choice([0, 1], 1)[0] == 0 else 'B'
        second_player = 'A' if first_player == 'B' else 'B'

        if action_a < 0 or action_a > 4 or action_b < 0 or action_b > 4:
            return self.get_state(), rewards, self.done

        # Player A moves first
        new_state = {}
        new_state[first_player] = self.player_positions[first_player] + action_diff[actions[first_player]]

        # If they collide, only the first play moves
        if (new_state[first_player] == self.player_positions[second_player]).all():
            #  If the player with the ball moves second, then the ball changes possession
            if self.possession == second_player:
                self.possession = first_player
            self.player_positions[first_player] = new_state[first_player]
            return self.get_state(), rewards, self.done
        new_state[second_player] = self.player_positions[second_player] + action_diff[actions[second_player]]

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

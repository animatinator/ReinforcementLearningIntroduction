# Windy gridworld SARSA solution

import numpy as np
import random
from windy_env import Action, TimeStep, WindyGridworld


EPSILON = 0.1


class QFunction:
	def __init__(self, width, height):
		self._q = np.zeros((height, width, len(Action)), dtype=np.float32)

	def get_value(self, state, action):
		return self._q[state.y][state.x][action.value]

	def set_value(elf, state, action, value):
		self._q[state.y][state.x][action.value] = value

	def optimal_action(self, state, possible_actions):
		# Absolute mess. Trying to get the maximum action from the possible actions.
		# First get the Q-values, and bind them with the actions they represent, ie.
		# (Q-value, Action).
		action_values = self._q[state[1]][state[0]]
		linked_to_actions = np.array([(val, Action(i)) for i, val in enumerate(action_values)])
		# Filter down to the actions that are possible.
		possibility_filter = np.isin(linked_to_actions[:, 1], list(possible_actions))
		possible_options = linked_to_actions[possibility_filter]
		# Return the action corresponding to the maximum Q-value.
		return possible_options[np.argmax(possible_options[:, 0])][1]


def e_greedy_action(state, possible_actions, q_function, epsilon):
	if (np.random.uniform() < epsilon):
		return random.sample(possible_actions)
	else:
		return q_function.optimal_action(state, possible_actions)


if __name__ == '__main__':
	winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
	height = 7
	goal_pos = (7, 3)

	env = WindyGridworld(winds, height, goal_pos)
	q = QFunction(width=len(winds), height=height)

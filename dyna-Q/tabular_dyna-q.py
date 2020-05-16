# Tabular Dyna-Q implementation
# Based on Sutton & Barto example 8.1.

from dataclasses import dataclass
from maze_env import Action, parse_maze_from_file
import numpy as np
import random


EPSILON = 0.1
DISCOUNT = 0.95
LEARNING_RATE = 0.2
TRAIN_STEPS = 10000
PLAN_STEPS = 1


class QFunction:
	def __init__(self, width, height):
		self._q = np.ones((height, width, len(Action)), dtype=np.float32)
	
	def get_value(self, state, action):
		return self._q[state[1]][state[0]][action.value]
	
	def set_value(self, state, action, value):
		self._q[state[1]][state[0]][action.value] = value

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
		return random.choice(list(possible_actions))
	else:
		return q_function.optimal_action(state, possible_actions)


@dataclass
class ModelEntry:
	state: (int, int)
	reward: float


class Model:
	def __init__(self, width, height):
		self._m = [[[None for action in Action] for x in range(width)] for y in range(height)]
	
	def get_value(self, state, action):
		return self._m[state[1]][state[0]][action.value]
	
	def set_value(self, state, action, new_state, reward):
		self._m[state[1]][state[0]][action.value] = ModelEntry(new_state, reward)


def train_and_evaluate(maze, train_steps, plan_steps):
	width, height = maze.dimensions()
	q = QFunction(width, height)
	model = Model(width, height)

	# Here we'll record the number of steps taken for each completed episode.
	steps_per_episode = []
	cur_episode_steps = 0

	state = maze.reset()

	for step in range(train_steps):
		action = e_greedy_action(state, maze.valid_actions(state), q, EPSILON)

		timestep = maze.step(state, action)
		cur_episode_steps += 1
		state_1 = timestep.state
		reward = timestep.reward

		q_s_a = q.get_value(state, action)
		action_1 = q.optimal_action(state_1, maze.valid_actions(state_1))
		q_s1_a1 = q.get_value(state_1, action_1)
		new_q_s_a = q_s_a + LEARNING_RATE * (reward + (DISCOUNT * q_s1_a1) - q_s_a)
		q.set_value(state, action, new_q_s_a)
		model.set_value(state, action, state_1, reward)

		for i in range(plan_steps):
			pass # TODO

		state = state_1

		if timestep.terminal:
			steps_per_episode.append(cur_episode_steps)
			state = maze.reset()
			cur_episode_steps = 0

	return steps_per_episode


if __name__ == '__main__':
	maze = parse_maze_from_file('maze.txt')
	steps_per_episode = train_and_evaluate(maze, TRAIN_STEPS, PLAN_STEPS)
	print(steps_per_episode)

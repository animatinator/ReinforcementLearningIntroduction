# Tabular Dyna-Q implementation
# Based on Sutton & Barto example 8.1.

from dataclasses import dataclass
import matplotlib.pyplot as plt
from maze_env import Action, parse_maze_from_file
import numpy as np
import pdb
import random


EPSILON = 0.1
DISCOUNT = 0.95
LEARNING_RATE = 0.2
TRAIN_STEPS = 10000
PLAN_STEPS = 10
NUM_RUNS = 3


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
		self._m = [[[ModelEntry((x, y), 0) for action in Action] for x in range(width)] for y in range(height)]
		self._w = width
		self._h = height
		self._visited = set()
		self._start_state = (0, 0)

	def get_start_state(self):
		return self._start_state

	def set_start_state(self, state):
		self._start_state = state

	def get_value(self, state, action):
		return self._m[state[1]][state[0]][action.value]

	def set_value(self, state, action, new_state, reward):
		self._visited.add(state)
		self._m[state[1]][state[0]][action.value] = ModelEntry(new_state, reward)

	def select_visited_s_a_pair(self):
		# Sadly, this is much the same as the awful optimal_action method in
		# QValue. Should really do this sort of thing better in future, but for
		# now I am hacking so copy-paste-debug is yes.
		state = random.sample(self._visited, 1)[0]
		action_models = self._m[state[1]][state[0]]
		linked_to_actions = np.array([(val, Action(i)) for i, val in enumerate(action_models) if val])
		action = random.choice(linked_to_actions)[1]
		return (state, action)

	# Select a potentially unvisited state-action pair.
	# (See constructor - unvisited state-action pairs are assumed to return to
	# the starting state with reward zero.)
	def select_s_a_pair(self):
		state = (random.choice(range(0, self._w)), random.choice(range(0, self._h)))
		action = random.choice([action for action in Action])
		return (state, action)


def train_and_evaluate(maze, train_steps, plan_steps, trajectory_sampling=False):
	width, height = maze.dimensions()
	q = QFunction(width, height)
	model = Model(width, height)

	# Here we'll record the number of steps taken for each completed episode.
	steps_per_episode = []
	cur_episode_steps = 0

	state = maze.reset().state
	model.set_start_state(state)

	for step in range(train_steps):
		action = e_greedy_action(state, maze.valid_actions(state), q, EPSILON)

		timestep = maze.step(state, action)
		cur_episode_steps += 1
		state_1 = timestep.state
		reward = timestep.reward

		# One-step Q-learning on this transition.
		q_s_a = q.get_value(state, action)
		action_1 = q.optimal_action(state_1, maze.valid_actions(state_1))
		q_s1_a1 = q.get_value(state_1, action_1)
		new_q_s_a = q_s_a + LEARNING_RATE * (reward + (DISCOUNT * q_s1_a1) - q_s_a)
		q.set_value(state, action, new_q_s_a)
		model.set_value(state, action, state_1, reward)

		plan_s = model.get_start_state()

		# Run plan_steps iterations of planning.
		# The naming scheme got wildly out of hand, sorry.
		for i in range(plan_steps):
			if trajectory_sampling:
				# Choose an action according to the current policy.
				plan_a = e_greedy_action(plan_s, maze.valid_actions(plan_s), q, EPSILON)
			else:
				# Pick a visited state and action.
				plan_s, plan_a = model.select_s_a_pair()

			# Ask the model what state and reward we'd get from that s-a pair.
			step = model.get_value(plan_s, plan_a)
			plan_s_1 = step.state
			plan_r = step.reward

			# One-step Q-learning on the modelled transition.
			plan_q_s_a = q.get_value(plan_s, plan_a)
			plan_a_1 = q.optimal_action(plan_s_1, maze.valid_actions(plan_s_1))
			plan_q_s1_a1 = q.get_value(plan_s_1, plan_a_1)
			new_plan_q_s_a = plan_q_s_a + LEARNING_RATE * (plan_r + (DISCOUNT * plan_q_s1_a1) - plan_q_s_a)
			q.set_value(plan_s, plan_a, new_plan_q_s_a)

			# Take the step so that trajectory-base sampling can continue.
			plan_s = plan_s_1

		state = state_1

		# Reset the state when we reach the end of an episode.
		if timestep.terminal:
			steps_per_episode.append(cur_episode_steps)
			state = maze.reset().state
			cur_episode_steps = 0

	return steps_per_episode


def trim_to_same_length(lists):
	min_length = len(min(lists, key=len))
	lists = [list[:min_length] for list in lists]
	return lists


def average_over_n_runs(fn, num_runs):
	steps_per_episode_records = []

	for run in range(num_runs):
		steps_per_episode_records.append(fn())

	steps_per_episode_records = trim_to_same_length(steps_per_episode_records)
	return [sum(values) / len(values) for values in zip(*steps_per_episode_records)]


def run_experiment(maze):
	steps_per_episode_records = []
	steps_per_episode_records.append(average_over_n_runs(lambda: train_and_evaluate(maze, TRAIN_STEPS, PLAN_STEPS), NUM_RUNS))
	steps_per_episode_records.append(average_over_n_runs(lambda: train_and_evaluate(maze, TRAIN_STEPS, PLAN_STEPS, trajectory_sampling=True), NUM_RUNS))

	# More successful methods will have completed more episodes. Cut them all
	# down to the lowest number of completed episodes for graph clarity.
	steps_per_episode_records = trim_to_same_length(steps_per_episode_records)
	xs = [i for i in range(len(steps_per_episode_records[0]))]

	for record in steps_per_episode_records:
		plt.plot(xs, record)

	plt.show()


if __name__ == '__main__':
	maze = parse_maze_from_file('maze.txt')
	run_experiment(maze)

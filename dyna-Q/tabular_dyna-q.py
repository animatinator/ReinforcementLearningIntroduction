# Tabular Dyna-Q implementation
# Based on Sutton & Barto example 8.1.

from dataclasses import dataclass
import matplotlib.pyplot as plt
from maze_env import Action, parse_maze_from_file
import numpy as np
import random


EPSILON = 0.1
DISCOUNT = 0.95
LEARNING_RATE = 0.2
TRAIN_STEPS = 2000
PLAN_STEPS_OPTIONS = [0, 5, 10, 50]


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
		self._visited = set()
	
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


def train_and_evaluate(maze, train_steps, plan_steps):
	width, height = maze.dimensions()
	q = QFunction(width, height)
	model = Model(width, height)

	# Here we'll record the number of steps taken for each completed episode.
	steps_per_episode = []
	cur_episode_steps = 0

	state = maze.reset().state

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

		# Run plan_steps iterations of planning.
		# The naming scheme got wildly out of hand, sorry.
		for i in range(plan_steps):
			# Pick a visited state and action.
			plan_s, plan_a = model.select_visited_s_a_pair()

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

		state = state_1

		# Reset the state when we reach the end of an episode.
		if timestep.terminal:
			steps_per_episode.append(cur_episode_steps)
			state = maze.reset().state
			cur_episode_steps = 0

	return steps_per_episode


# For a list of planning_steps values, train and evaluate models in the maze
# for each number of planning steps and graph their time taken to complete each
# successive episode over time.
def compare_planning_levels(maze, levels):
	steps_per_episode_records = []
	for plan_steps in levels:
		steps_per_episode = train_and_evaluate(maze, TRAIN_STEPS, plan_steps)
		steps_per_episode_records.append(steps_per_episode)

	# More successful model-based methods will have completed more episodes.
	# Cut them all down to the lowest number of completed episodes for graph
	# clarity.
	min_length = len(min(steps_per_episode_records, key=len))
	steps_per_episode_records = [record[:min_length] for record in steps_per_episode_records]
	xs = [i for i in range(min_length)]

	for record in steps_per_episode_records:
		plt.plot(xs, record)

	plt.show()


if __name__ == '__main__':
	maze = parse_maze_from_file('maze.txt')
	compare_planning_levels(maze, PLAN_STEPS_OPTIONS)

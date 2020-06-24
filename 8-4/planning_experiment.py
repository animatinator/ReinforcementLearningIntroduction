# Experimenting with tabulary dyna-Q planning using two different methods to
# deal with changes in the environment.

from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
from maze_env import Action, parse_maze_from_file, MazeEnvironment, DynamicMazeEnvironment
import numpy as np
import random


EPSILON = 0.1
DISCOUNT = 0.95
LEARNING_RATE = 0.2
TRAIN_STEPS = 2000
PLAN_STEPS = 20
INFREQUENT_REWARD_SCALING_FACTOR = 0.005

MAZE_UNBLOCKED_FILE = 'maze_unblocked.txt'
MAZE_BLOCKED_FILE = 'maze.txt'


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
	last_visit: int


class Model:
	def __init__(self, width, height, amplify_infrequent_rewards = False):
		self._w = width
		self._h = height
		self._m = [[[ModelEntry((x, y), 0, 0) for action in Action] for x in range(width)] for y in range(height)]
		self._visited = set()
		self._time = 0
		self._amplify_infrequent_rewards = amplify_infrequent_rewards
	
	def get_value(self, state, action):
		step = self._m[state[1]][state[0]][action.value]
		if self._amplify_infrequent_rewards:
			step.reward += INFREQUENT_REWARD_SCALING_FACTOR * math.sqrt(float(self._time - step.last_visit))
		return step
	
	def set_value(self, state, action, new_state, reward):
		self._visited.add(state)
		self._m[state[1]][state[0]][action.value] = ModelEntry(new_state, reward, self._time)
		self._time += 1

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


def train_and_evaluate(maze, train_steps, plan_steps, amplify_infrequent_rewards=False):
	width, height = maze.dimensions()
	q = QFunction(width, height)
	model = Model(width, height, amplify_infrequent_rewards)

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
			# If we're using the modified algorithm that boosts rewards for
			# states not recently visited, pick /any/ state-action pair.
			plan_s, plan_a = model.select_s_a_pair() if amplify_infrequent_rewards else model.select_visited_s_a_pair()

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


def evaluate_and_plot(maze):
	plots = []

	# Normal Dyna-Q.
	plots.append(train_and_evaluate(maze, TRAIN_STEPS, PLAN_STEPS, False))
	maze.reset_dynamics()
	# Amplify rewards on states not visited in some time and allow planning to
	# consider unvisited state-action pairs.
	plots.append(train_and_evaluate(maze, TRAIN_STEPS, PLAN_STEPS, True))

	min_length = len(min(plots, key=len))
	plots = [record[:min_length] for record in plots]
	xs = [i for i in range(min_length)]

	for record in plots:
		plt.plot(xs, record)
	plt.show()


if __name__ == '__main__':
	unblocked_layout = parse_maze_from_file(MAZE_UNBLOCKED_FILE)
	blocked_layout = parse_maze_from_file(MAZE_BLOCKED_FILE)
	maze = DynamicMazeEnvironment([(0, unblocked_layout), (20, blocked_layout)])
	evaluate_and_plot(maze)

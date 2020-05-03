# Windy gridworld SARSA solution

import numpy as np
import random
from windy_env import Action, KINGS_ACTIONS, STANDARD_ACTIONS, TimeStep, WindyGridworld


EPSILON = 0.05
ALPHA = 0.1
TRAIN_STEPS = 100000
REPORT_EVERY = 1000

WINDS = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
WIDTH = len(WINDS)
HEIGHT = 7
GOAL_POS = (7, 3)
START_STATE = (0, 3)


class QFunction:
	def __init__(self, width, height, action_set):
		self._q = np.zeros((height, width, len(action_set)), dtype=np.float32)

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


if __name__ == '__main__':
	finished_episodes = 0

	env = WindyGridworld(WINDS, HEIGHT, GOAL_POS, KINGS_ACTIONS)
	q = QFunction(WIDTH, HEIGHT, KINGS_ACTIONS)

	state = START_STATE
	action = e_greedy_action(state, env.available_actions(state), q, EPSILON)

	for i in range(TRAIN_STEPS):
		# Every REPORT_EVERY steps, print out how many times we reached the
		# goal since the last report and reset the count.
		if (i % REPORT_EVERY == 0):
			print("Step {}, {} episodes completed since last report.".format(i, finished_episodes))
			finished_episodes = 0

		# Step, note the reward and get the next state and action.
		timestep = env.step(state, action)
		reward = timestep.reward
		state_1 = timestep.state
		action_1 = e_greedy_action(state_1, env.available_actions(state_1), q, EPSILON)

		# Update Q(state, action) based on Q(state', action')
		q_s_a = q.get_value(state, action)
		q_s1_a1 = q.get_value(state_1, action_1)
		# No discounting here.
		new_q_s_a = q_s_a + ALPHA * (reward + q_s1_a1 - q_s_a)
		q.set_value(state, action, new_q_s_a)

		state = state_1
		action = action_1

		# Reset and increment finished_episodes if we reached the goal.
		if timestep.terminal:
			finished_episodes += 1
			state = START_STATE
			action = e_greedy_action(state, env.available_actions(state), q, EPSILON)

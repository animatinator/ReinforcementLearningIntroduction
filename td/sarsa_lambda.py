# Windy gridworld TD(lambda) solution (trying out eligibility traces).

import numpy as np
import random
from windy_env import Action, EXTENDED_KINGS_ACTIONS, TimeStep, WindyGridworld


ALPHA = 0.3
LAMBDA = 0.8
EPSILON = 0.05
DISCOUNT = 0.95
TRAIN_STEPS = 50000
REPORT_EVERY = 1000

WINDS = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
WIDTH = len(WINDS)
HEIGHT = 7
GOAL_POS = (7, 3)
START_STATE = (0, 3)

TILE_OFFSETS = [(0.0, 0.0, 0.0), (-0.1, -0.3, -0.5), (-0.5, -0.7, -0.9), (-0.45, -0.1, -0.32), (-0.8, -0.23, -0.5)]


# A 3D function sampler and a sample 3D function to test the tiling approximator.
def sample_3d_fn(domain, fn):
	x = random.random() * float(domain[0])
	y = random.random() * float(domain[1])
	z = random.random() * float(domain[2])
	val = fn(x, y, z)
	return (x, y, z, val)

def test_3d_fn(domain):
	w, h, d = domain
	def fn(x, y, z):
		if x > (w / 2) and y > (h / 2) and z > (d / 2):
			return 1.0
		return 0.0
	return fn


# An approximator for 3D functions that uses a series of offset tilings with individual weights.
class Tiling3D:
	def __init__(self, domain, tile_dimensions, tile_offsets):
		assert len(tile_offsets) > 0, "Must specify at least one tile offset"
		assert len(domain) == 3, "Domain must specify three axes"
		assert len(tile_dimensions) == 3, "Must specify tile size in three axes"
		# It's more convenient to assume tilings all start /before/ the
		# beginning of the range
		assert all(all(dim <= 0 for dim in elem) for elem in tile_offsets), "All tile offsets must be <= 0"

		self._domain = domain
		self._tile_dimensions = tile_dimensions
		self._offsets = [(x * tile_dimensions[0], y * tile_dimensions[1], z * tile_dimensions[2]) for (x, y, z) in tile_offsets]
		# Per the above assumption, add an extra tile to the end to ensure we
		# cover the whole range.
		self._num_tiles_per_tiling = (int(float(domain[i]) / tile_size) + 1 for i, tile_size in enumerate(tile_dimensions))
		# Set up a T*Nx*Ny array of weights where T = the number of different
		# tilings and N = the numer of tiles in each tiling. Nx is the x axis
		# and Ny the y.
		self._weights = np.zeros((len(tile_offsets), *self._num_tiles_per_tiling), np.float64)
		self._trace = np.copy(self._weights)
	
	def _check_in_domain(self, x, y, z):
		assert 0.0 <= x <= self._domain[0], f"X must be in the range [0, {self._domain[0]}]"
		assert 0.0 <= y <= self._domain[1], f"Y must be in the range [0, {self._domain[1]}]"
		assert 0.0 <= z <= self._domain[2], f"Z must be in the range [0, {self._domain[2]}]"

	def sample(self, x, y, z):
		self._check_in_domain(x, y, z)

		result = 0

		for i, offset in enumerate(self._offsets):
			tile_x = int((x - offset[0]) / self._tile_dimensions[0])
			tile_y = int((y - offset[1]) / self._tile_dimensions[1])
			tile_z = int((z - offset[2]) / self._tile_dimensions[2])
			result += self._weights[i, tile_x, tile_y, tile_z]

		return result

	def _action_values(self, x, y):
		self._check_in_domain(x, y, 0)

		values = [self.sample(x, y, a) for a in range(self._domain[2])]

		return values

	def learn_from_sample(self, x, y, z, value):
		self._check_in_domain(x, y, z)

		rate = ALPHA / float(len(self._offsets))

		for i, offset in enumerate(self._offsets):
			tile_x = int((x - offset[0]) / self._tile_dimensions[0])
			tile_y = int((y - offset[1]) / self._tile_dimensions[1])
			tile_z = int((z - offset[2]) / self._tile_dimensions[2])
			self._weights[i, tile_x, tile_y, tile_z] += rate * (value - self.sample(x, y, z))

	def update_trace_and_get_delta_for_SA(self, state, action):
		x, y = state
		z = action.value
		delta = 0

		for i, offset in enumerate(self._offsets):
			tile_x = int((x - offset[0]) / self._tile_dimensions[0])
			tile_y = int((y - offset[1]) / self._tile_dimensions[1])
			tile_z = int((z - offset[2]) / self._tile_dimensions[2])
			delta -= self._weights[i, tile_x, tile_y, tile_z]
			self._trace[i, tile_x, tile_y, tile_z] += 1

		return delta

	def get_updated_delta_for_end_of_step(self, state, action, delta):
		x, y = state
		z = action.value

		for i, offset in enumerate(self._offsets):
			tile_x = int((x - offset[0]) / self._tile_dimensions[0])
			tile_y = int((y - offset[1]) / self._tile_dimensions[1])
			tile_z = int((z - offset[2]) / self._tile_dimensions[2])
			delta += DISCOUNT * self._weights[i, tile_x, tile_y, tile_z]

		return delta

	def decay_trace(self):
		rate = ALPHA / float(len(self._offsets))
		self._trace *= rate * DISCOUNT

	def update_from_trace(self, delta):
		rate = ALPHA / float(len(self._offsets))
		self._weights += rate * delta * self._trace

	def optimal_action(self, x, y, possible_actions):
		# Absolute mess. Trying to get the maximum action from the possible actions.
		# First get the Q-values, and bind them with the actions they represent, ie.
		# (Q-value, Action).
		action_values = self._action_values(x, y)
		linked_to_actions = np.array([(val, Action(i)) for i, val in enumerate(action_values)])
		# Filter down to the actions that are possible.
		possibility_filter = np.isin(linked_to_actions[:, 1], list(possible_actions))
		possible_options = linked_to_actions[possibility_filter]
		# Return the action corresponding to the maximum Q-value.
		return possible_options[np.argmax(possible_options[:, 0])][1]


def e_greedy_action(state, possible_actions, value_function, epsilon):
	if (np.random.uniform() < epsilon):
		return random.choice(list(possible_actions))
	else:
		return value_function.optimal_action(state[0], state[1], possible_actions)


def train_3d_approximation_for_test(domain, approximation, fn):
	for i in range(TRAIN_STEPS):
		x, y, z, value = sample_3d_fn(domain, fn)
		approximation.learn_from_sample(x, y, z, value)


if __name__ == '__main__':
	finished_episodes = 0

	env = WindyGridworld(
		WINDS, HEIGHT, GOAL_POS, EXTENDED_KINGS_ACTIONS, stochastic_wind = True)
	domain = (WIDTH + 1, HEIGHT + 1, len(EXTENDED_KINGS_ACTIONS))
	tiling = Tiling3D(domain=domain, tile_dimensions=(2.0, 2.0, 2.0), tile_offsets=TILE_OFFSETS)

	state = START_STATE
	action = e_greedy_action(state, env.available_actions(state), tiling, EPSILON)

	for i in range(TRAIN_STEPS):
		# Every REPORT_EVERY steps, print out how many times we reached the
		# goal since the last report and reset the count.
		if (i % REPORT_EVERY == 0):
			print("Step {}, {} episodes completed since last report.".format(i, finished_episodes))
			finished_episodes = 0

		# Step, note the reward and get the next state and action.
		timestep = env.step(state, action)
		delta = timestep.reward
		state_1 = timestep.state

		delta += tiling.update_trace_and_get_delta_for_SA(state, action)

		# Reset and increment finished_episodes if we reached the goal.
		if timestep.terminal:
			tiling.update_from_trace(delta)
			finished_episodes += 1
			state = START_STATE
			action = e_greedy_action(state, env.available_actions(state), tiling, EPSILON)

		action_1 = e_greedy_action(state_1, env.available_actions(state_1), tiling, EPSILON)
		delta = tiling.get_updated_delta_for_end_of_step(state_1, action_1, delta)
		tiling.update_from_trace(delta)
		tiling.decay_trace()
		state = state_1
		action = action_1

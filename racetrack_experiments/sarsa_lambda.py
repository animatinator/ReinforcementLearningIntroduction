# Windy gridworld TD(lambda) solution (trying out eligibility traces).

import numpy as np
import random
from racetrack import constants
from racetrack.policy import Policy
from racetrack.track import Action, TrackEnvironment, read_track
from save_policy import save_policy


ALPHA = 0.3
LAMBDA = 0.8
EPSILON = 0.05
DISCOUNT = 0.95
TRAIN_STEPS = 50000
REPORT_EVERY = 1000

TILE_OFFSETS_3D = [(0.0, 0.0, 0.0),
	(-0.1, -0.3, -0.5),
	(-0.5, -0.7, -0.9),
	(-0.45, -0.1, -0.32),
					(-0.8, -0.23, -0.5)]
TILE_OFFSETS_4D = [(0.0, 0.0, 0.0, 0.0),
	(-0.1, -0.3, -0.5, -0.7),
	(-0.5, -0.7, -0.9, -0.1),
	(-0.45, -0.1, -0.32, -0.27),
	(-0.8, -0.23, -0.5, -0.73)]
TILE_OFFSETS_5D = [(0.0, 0.0, 0.0, 0.0, 0.0),
	(-0.1, -0.3, -0.5, -0.7, -0.9),
	(-0.5, -0.7, -0.9, -0.1, -0.3),
	(-0.45, -0.1, -0.32, -0.27, -0.51),
	(-0.8, -0.23, -0.5, -0.73, -0.16)]


# A 4D function sampler and a sample 4D function to test the tiling approximator.
def sample_nd_fn(domain, fn):
	point = [random.random() * float(domain[d]) for d in range(0, len(domain))]
	val = fn(point)
	return (point, val)

def test_nd_fn(domain):
	def fn(point):
		for d in range(0, len(domain)):
			if point[d] < (domain[d] / 2):
				return 0.0
		return 1.0
	return fn


# An approximator for n-dimensional functions that uses a series of offset tilings with individual weights.
class CoarseTiling:
	def __init__(self, domain, tile_dimensions, tile_offsets):
		assert len(tile_offsets) > 0, "Must specify at least one tile offset"
		assert len(domain) == len(tile_dimensions), "Domain and tile_dimensions must have the same dimension"
		# It's more convenient to assume tilings all start /before/ the
		# beginning of the range
		assert all(all(dim <= 0 for dim in elem) for elem in tile_offsets), "All tile offsets must be <= 0"

		self._dimension = len(domain)
		self._domain = domain
		self._tile_dimensions = tile_dimensions
		self._original_tile_offsets = tile_offsets
		self._offsets = [[x * tile_dimensions[d] for d, x in enumerate(offsets)] for offsets in tile_offsets]
		# Per the above assumption, add an extra tile to the end to ensure we
		# cover the whole range.
		self._num_tiles_per_tiling = [int(float(domain[i]) / tile_size) + 1 for i, tile_size in enumerate(tile_dimensions)]
		# Set up a T*Nx*Ny*... array of weights where T = the number of different
		# tilings and N = the numer of tiles in each tiling. Nx is the x axis
		# and Ny the y etc.
		self._weights = np.zeros((len(tile_offsets), *self._num_tiles_per_tiling), np.float64)
		# Set up an elegibility trace to track the eligibility of each weight
		# for the current update.
		self._trace = np.copy(self._weights)
	
	def _check_in_domain(self, point):
		for i, limit in enumerate(self._domain):
			assert 0.0 <= point[i] <= limit

	def _get_offsets_for_point_in_ith_tiling(self, point, i):
		assert 0 <= i < len(self._offsets), f"Tiling {i} not in range - only {len(self._offsets)} tilings defined."

		tiling_offset = self._offsets[i]
		position = [int((point[d] - tiling_offset[d]) / self._tile_dimensions[d]) for d in range(0, self._dimension)]
		return position

	# Evaluate the approximation.
	def _sample(self, point):
		self._check_in_domain(point)

		result = 0

		for i in range(0, len(self._offsets)):
			position = self._get_offsets_for_point_in_ith_tiling(point, i)
			result += self._weights[(i, *position)]

		return result

	# Compute the values of each of the actions at a given position.
	def _action_values(self, point):
		assert len(point) == self._dimension - 1, f"Point should be {self._dimension - 1}-dimensional, but is {len(point)}-dimensional."
		self._check_in_domain((*point, 0))

		values = [self._sample((*point, a)) for a in range(self._domain[-1])]

		return values

	def update_trace_and_get_delta_for_step_start(self, state, action):
		assert len(state) == self._dimension - 1, f"State should be {self._dimension - 1}-dimensional, but is {len(state)}-dimensional."
		delta = 0

		for i in range(0, len(self._offsets)):
			position = self._get_offsets_for_point_in_ith_tiling((*state, action), i)
			delta -= self._weights[(i, *position)]
			self._trace[(i, *position)] += 1

		return delta

	def get_updated_delta_for_end_of_step(self, state, action, delta):
		assert len(state) == self._dimension - 1, f"State should be {self._dimension - 1}-dimensional, but is {len(state)}-dimensional."

		for i in range(0, len(self._offsets)):
			position = self._get_offsets_for_point_in_ith_tiling((*state, action), i)
			delta += DISCOUNT * self._weights[(i, *position)]

		return delta

	def decay_trace_after_step(self):
		rate = ALPHA / float(len(self._offsets))
		self._trace *= rate * DISCOUNT

	def update_from_delta_using_current_trace(self, delta):
		rate = ALPHA / float(len(self._offsets))
		self._weights += rate * delta * self._trace

	def optimal_action(self, point, possible_actions):
		# Absolute mess. Trying to get the maximum action from the possible actions.
		# First get the Q-values, and bind them with the actions they represent, ie.
		# (Q-value, Action).
		action_values = self._action_values(point)
		linked_to_actions = np.array([(val, Action(i)) for i, val in enumerate(action_values)])
		# Filter down to the actions that are possible.
		possibility_filter = np.isin(linked_to_actions[:, 1], list(possible_actions))
		possible_options = linked_to_actions[possibility_filter]
		# Return the action corresponding to the maximum Q-value.
		return possible_options[np.argmax(possible_options[:, 0])][1]

	def learn_from_sample(self, point, value):
		self._check_in_domain(point)

		rate = ALPHA / float(len(self._offsets))

		for i in range(0, len(self._offsets)):
			position = self._get_offsets_for_point_in_ith_tiling(point, i)
			self._weights[(i, *position)] += rate * (value - self._sample(point))

	def serialise(self):
		return {
			'domain': self._domain,
			'tile_dimensions': self._tile_dimensions,
			'tile_offsets': self._original_tile_offsets,
			'weights': self._weights
		}

	def override_weights_from_serialisation(self, weights):
		self._weights = weights

def deserialise_coarse_tiling(serialised_dict):
	tiling = CoarseTiling(serialised_dict['domain'], serialised_dict['tile_dimensions'], serialised_dict['tile_offsets'])
	tiling.override_weights_from_serialisation(serialised_dict['weights'])
	return tiling


def vectorise_state(state):
	return (state.pos[0], state.pos[1], state.vel[0], state.vel[1])


def e_greedy_action(state, possible_actions, value_function, epsilon):
	if (np.random.uniform() < epsilon):
		return random.choice(list(possible_actions))
	else:
		return value_function.optimal_action(vectorise_state(state), possible_actions)


def train_nd_approximation_for_test(domain, approximation, fn):
	for i in range(TRAIN_STEPS):
		point, value = sample_nd_fn(domain, fn)
		approximation.learn_from_sample(point, value)


class ApproximatedPolicy(Policy):
	def __init__(self, approximator):
		self._v = approximator

	def get_action(self, state, possible_actions):
		return self._v.optimal_action(vectorise_state(state), possible_actions)


class EGreedyApproximatedPolicy(ApproximatedPolicy):
	def get_action(self, state, possible_actions):
		return e_greedy_action(state, possible_actions, self._v, EPSILON)


def train_and_evaluate():
	finished_episodes = 0

	track = read_track('racetrack/track.bmp')
	env = TrackEnvironment(track)
	size = env.size()
	domain = (size[0] + 1, size[1] + 1, constants.MAX_VELOCITY + 1, constants.MAX_VELOCITY + 1, len(Action))

	tiling = CoarseTiling(domain, tile_dimensions=(2.5, 2.5, 2.0, 2.0, 1.0), tile_offsets = TILE_OFFSETS_5D)

	state = env.reset().state
	action = e_greedy_action(state, env.get_available_actions(state), tiling, EPSILON)

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

		delta += tiling.update_trace_and_get_delta_for_step_start(vectorise_state(state), action.value)

		# Reset and increment finished_episodes if we reached the goal.
		if timestep.terminal:
			tiling.update_from_delta_using_current_trace(delta)
			finished_episodes += 1
			state = env.reset().state
			action = e_greedy_action(state, env.get_available_actions(state), tiling, EPSILON)
			continue

		action_1 = e_greedy_action(state_1, env.get_available_actions(state_1), tiling, EPSILON)
		delta = tiling.get_updated_delta_for_end_of_step(vectorise_state(state_1), action_1.value, delta)
		tiling.update_from_delta_using_current_trace(delta)
		tiling.decay_trace_after_step()
		state = state_1
		action = action_1
	
	print("Saving policy...")
	save_policy(tiling.serialise())
	print("Done!")



if __name__ == '__main__':
	train_and_evaluate()

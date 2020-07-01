# Windy gridworld TD(lambda) solution (trying out eligibility traces).

import numpy as np
import random
from windy_env import Action, EXTENDED_KINGS_ACTIONS, TimeStep, WindyGridworld


ALPHA = 0.3
LAMBDA = 0.8
TRAIN_STEPS = 500000
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
		self._num_tiles_per_tiling = (int(float(domain[i] + 1) / tile_size) + 1 for i, tile_size in enumerate(tile_dimensions))
		# Set up a T*Nx*Ny array of weights where T = the number of different
		# tilings and N = the numer of tiles in each tiling. Nx is the x axis
		# and Ny the y.
		self._weights = np.zeros((len(tile_offsets), *self._num_tiles_per_tiling), np.float64)
	
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

	def learn_from_sample(self, x, y, z, value):
		self._check_in_domain(x, y, z)

		rate = ALPHA / float(len(self._offsets))

		for i, offset in enumerate(self._offsets):
			tile_x = int((x - offset[0]) / self._tile_dimensions[0])
			tile_y = int((y - offset[1]) / self._tile_dimensions[1])
			tile_z = int((z - offset[2]) / self._tile_dimensions[2])
			self._weights[i, tile_x, tile_y, tile_z] += rate * (value - self.sample(x, y, z))


def train_3d_approximation_for_test(domain, approximation, fn):
	for i in range(TRAIN_STEPS):
		x, y, z, value = sample_3d_fn(domain, fn)
		approximation.learn_from_sample(x, y, z, value)


if __name__ == '__main__':
	domain = (WIDTH, HEIGHT, len(EXTENDED_KINGS_ACTIONS))
	tiling = Tiling3D(domain=domain, tile_dimensions=(2.0, 2.0, 2.0), tile_offsets=TILE_OFFSETS)
	train_3d_approximation_for_test(domain, tiling, test_3d_fn(domain))
	print(tiling.sample(1, 1, 1))
	print(tiling.sample(6, 6, 6))
	print(tiling.sample(6, 4, 4))

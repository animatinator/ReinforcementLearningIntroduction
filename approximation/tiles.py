# Tile-based function approximation.

import functions
import matplotlib.pyplot as plt
import numpy as np


ALPHA = 0.2
TRAIN_STEPS = 10000
PLOT_RESOLUTION = 200


class Tiling1D:
	def __init__(self, tile_width, tile_offsets):
		assert len(tile_offsets) > 0, "Must specify at least one tile offset"
		# It's more convenient to assume tilings all start /before/ the
		# beginning of the range
		assert all(elem <= 0 for elem in tile_offsets), "All tile offsets must be <= 0"

		self._tile_width = tile_width
		self._offsets = tile_offsets
		# Per the above assumption, add an extra tile to the end to ensure we
		# cover the whole range.
		self._num_tiles_per_tiling = int(1.0 / tile_width) + 1
		# Set up a T*N array of weights where T = the number of different
		# tilings and N = the numer of tiles in each tiling.
		self._weights = np.zeros((len(tile_offsets), self._num_tiles_per_tiling), np.float64)
	
	def sample(self, x):
		assert 0.0 <= x <= 1.0, "X must be in the range [0, 1]"
		
		result = 0
		
		for i, offset in enumerate(self._offsets):
			tile_index = int((x - offset) / self._tile_width)
			result += self._weights[i, tile_index]
		
		return result
	
	def learn_from_sample(self, x, value):
		assert 0.0 <= x <= 1.0, "X must be in the range [0, 1]"
		
		rate = ALPHA / float(len(self._offsets))
		
		for i, offset in enumerate(self._offsets):
			tile_index = int((x - offset) / self._tile_width)
			self._weights[i, tile_index] += rate * (value - self._weights[i, tile_index])


def train_approximation(approximation, fn):
	for i in range(TRAIN_STEPS):
		x, value = functions.sample(fn)
		approximation.learn_from_sample(x, value)


def graph_approximation(approximation, resolution):
	xs = [float(i) / float(resolution) for i in range(resolution + 1)]
	values = [approximation.sample(x) for x in xs]
	plt.plot(xs, values)
	plt.show()


if __name__ == '__main__':
	tiling = Tiling1D(0.2, [0.0, -0.1, -0.15, -0.04, -0.12])
	train_approximation(tiling, functions.sine_1d)
	graph_approximation(tiling, PLOT_RESOLUTION)
	#tiling = Tiling1D(0.2, [0.0, -0.1, -0.15])

# Functions to approximate.
# All are defined in the range [0, 1] in each dimension.

import math
import random

def sample(fn):
	x = random.random()
	val = fn(x)
	return (x, val)

def sine_1d(x):
	return math.sin(x * 2* math.pi)

def step_1d(x):
	if x > 0.25 and x < 0.75:
		return 1.0
	return 0.0

# Functions to approximate.
# All are defined in the range [0, 1] in each dimension.

import math
import random

def sample_1d(fn):
	x = random.random()
	val = fn(x)
	return (x, val)

def sample_2d(fn):
	x = random.random()
	y = random.random()
	val = fn(x, y)
	return (x, y, val)

def sine_1d(x):
	return math.sin(x * 2* math.pi)

def step_1d(x):
	if x > 0.25 and x < 0.75:
		return 1.0
	return 0.0

def sum_2d(x, y):
	return x + y

def step_2d(x, y):
	if x > 0.25 and x < 0.75 and y > 0.25 and y < 0.75:
		return 1.0
	return 0.0

def ripple_2d(x, y):
	dist = math.sqrt(math.pow(x - 0.5, 2.0) + math.pow(y - 0.5, 2.0))
	return sine_1d(dist * 3.0)

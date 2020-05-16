# Tabular Dyna-Q implementation
# Based on Sutton & Barto example 8.1.

from dataclasses import dataclass
from maze_env import Action, parse_maze_from_file
import numpy as np


class QFunction:
	def __init__(self, width, height):
		self._q = np.zeros((height, width, len(Action)), dtype=np.float32)
	
	def get_value(self, state, action):
		return self._q[state[1]][state[0]][action.value]
	
	def set_value(self, state, action, value):
		self._q[state[1]][state[0]][action.value] = value


@dataclass
class ModelEntry:
	state: (int, int)
	reward: float


class Model:
	def __init__(self, width, height):
		self._m = [[[None for action in Action] for x in range(width)] for y in range(height)]
	
	def get_value(self, state, action):
		return self._m[state[1]][state[0]][action.value]
	
	def set_value(self, state, action, new_state, reward):
		self._m[state[1]][state[0]][action.value] = ModelEntry(new_state, reward)


if __name__ == '__main__':
	maze = parse_maze_from_file('maze.txt')
	width, height = maze.dimensions()
	q = QFunction(width, height)
	model = Model(width, height)

# A simple 2D maze environment.
# The maze is parsed from a text file.

from dataclasses import dataclass
from enum import Enum, unique
import numpy as np
import os


@dataclass
class TimeStep:
	state: (int, int)
	reward: float
	terminal: bool = False


@unique
class Action(Enum):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3


class MazeEnvironment:
	def __init__(self, layout, start_pos, goal_pos, random_rewards=False):
		self._grid = layout
		self._s = start_pos
		self._g = goal_pos
		self._w = len(self._grid[0])
		self._h = len(self._grid)
		self._random_rewards = random_rewards

	def dimensions(self):
		return (self._w, self._h)

	def reset(self):
		return TimeStep(self._s, 0.0)

	def _is_space(self, state):
		in_bounds = state[0] >= 0 and state[1] >= 0 and state[0] < self._w  and state[1] < self._h
		return in_bounds and not self._grid[state[1]][state[0]]

	def _apply_action(self, state, action):
		if action == Action.UP:
			return (state[0], state[1] - 1)
		elif action == Action.DOWN:
			return (state[0], state[1] + 1)
		elif action == Action.LEFT:
			return (state[0] - 1, state[1])
		elif action == Action.RIGHT:
			return (state[0] + 1, state[1])
		else:
			raise ValueError('Unknown action: "{}"'.format(action))

	def valid_actions(self, state):
		actions = set()
		for action in Action:
			if self._is_space(self._apply_action(state, action)):
				actions.add(action)
		return actions

	def step(self, state, action):
		# If we're already at the goal, return the start state with zero reward.
		if state == self._g:
			return TimeStep(self._s, 0.0, False)

		assert(action in self.valid_actions(state))
		
		state = self._apply_action(state, action)
		if state == self._g:
			return TimeStep(state, 1.0, True)
		else:
			reward = np.random.normal(scale=0.2) if self._random_rewards else 0.0
			return TimeStep(state, reward)

	def set_random_rewards(self, should_use):
		self._random_rewards = should_use


def parse_maze_from_file(file_path):
	layout = []
	start = (-1, -1)
	goal = (-1, -1)

	infile = open(os.path.join(os.path.dirname(__file__), file_path));
	for y, line in enumerate(infile.readlines()):
		row = []
		for x, col in enumerate(line):
			if col == '.':
				row.append(False)
			elif col == 'X':
				row.append(True)
			elif col == 'S':
				start = (x, y)
				row.append(False)
			elif col == 'G':
				goal = (x, y)
				row.append(False)
			elif col == '\n':
				pass
			else:
				raise ValueError('Unexpected character in maze definition file: "{}"'.format(col))
		layout.append(row)

	assert(start != (-1, -1))
	assert(goal != (-1, -1))

	return MazeEnvironment(layout, start, goal)

	
if __name__ == '__main__':
	maze = parse_maze_from_file("maze.txt")

	assert(maze.reset() == (0, 2))

	assert(maze.valid_actions((0, 0)) == set([Action.DOWN, Action.RIGHT]))
	assert(maze.valid_actions((8, 1)) == set([Action.UP, Action.DOWN]))
	assert(maze.valid_actions((4, 4)) == set([Action.UP, Action.DOWN, Action.LEFT]))

	assert(maze.step((4, 4), Action.LEFT) == TimeStep((3, 4), 0.0, False))
	assert(maze.step((8, 1), Action.UP) == TimeStep((8, 0), 1.0, True))

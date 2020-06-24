# A simple 2D maze environment.
# The maze is parsed from a text file.

from dataclasses import dataclass
from enum import Enum, unique
import os
from typing import List


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


@dataclass
class MazeLayout:
	grid: List[List[bool]]
	start: (int, int)
	goal: (int, int)
	
	def dimensions(self):
		return(len(self.grid[0]), len(self.grid[1]))
	
	def in_bounds(self, pos):
		return pos[0] >= 0 and pos[1] >= 0 and pos[0] < len(self.grid[0]) and pos[1] < len(self.grid)


class MazeEnvironment:
	def __init__(self, layout):
		self._layout = layout

	def dimensions(self):
		return self._layout.dimensions()

	def reset(self):
		return TimeStep(self._layout.start, 0.0)

	def _is_space(self, state):
		return self._layout.in_bounds(state) and not self._layout.grid[state[1]][state[0]]

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
		if state == self._layout.goal:
			return TimeStep(self._layout.start, 0.0, False)

		assert(action in self.valid_actions(state))
		
		state = self._apply_action(state, action)
		if state == self._layout.goal:
			return TimeStep(state, 1.0, True)
		else:
			return TimeStep(state, 0.0)


# A dynamic maze whose layout can change after specified numbers of episodes.
# It takes a list of layout keyframes [(int, MazeLayout)] with each keyframe
# specifying the duration to wait after the previous keyframe and the layout
# to switch to. 'Duration' is measured in resets. The first duration must be
# zero (and is sanity-checked).
#
# For example, [(0, easy_layout), (10, hard_layout), (20, easy_layout)] will
# start with easy_layout, then the tenth reset() call will switch it to
# hard_layout. After a further twenty resets, it will switch back to
# easy_layout.
class DynamicMazeEnvironment(MazeEnvironment):
	def __init__(self, layout_keyframes):
		assert len(layout_keyframes) > 1, "Must have at least one keyframe."
		assert layout_keyframes[0][0] == 0, "First keyframe must have a wait duration of zero."

		super().__init__(layout_keyframes[0][1])
		self._keyframes = layout_keyframes[1:]
		self.frame_counter = 0
	
	def reset(self):
		self.frame_counter += 1
		if self._keyframes and self.frame_counter == self._keyframes[0][0]:
			self._layout = self._keyframes[0][1]
			self._keyframes = self._keyframes[1:]
			self.frame_counter = 0
		return super().reset()


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

	return MazeLayout(layout, start, goal)

	
if __name__ == '__main__':
	maze = MazeEnvironment(parse_maze_from_file("maze.txt"))

	assert(maze.reset().state == (9, 4))

	assert(maze.valid_actions((0, 0)) == set([Action.DOWN, Action.RIGHT]))
	assert(maze.valid_actions((0, 2)) == set([Action.UP, Action.DOWN]))
	assert(maze.valid_actions((6, 1)) == set([Action.UP, Action.LEFT, Action.RIGHT]))

	assert(maze.step((4, 4), Action.LEFT) == TimeStep((3, 4), 0.0, False))
	assert(maze.step((9, 1), Action.UP) == TimeStep((9, 0), 1.0, True))
	
	dynamic_maze = DynamicMazeEnvironment([(0, parse_maze_from_file("maze.txt")), (1, parse_maze_from_file("maze_unblocked.txt"))])

	assert(dynamic_maze.valid_actions((9, 3)) == set([Action.DOWN, Action.LEFT]))

	# Reset and verify that the layout changed.
	assert(dynamic_maze.reset().state == (9, 4))
	assert(dynamic_maze.valid_actions((9, 3)) == set([Action.UP, Action.DOWN, Action.LEFT]))
	
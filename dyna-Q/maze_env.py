# A simple 2D maze environment.
# The maze is parsed from maze.txt.

from enum import Enum
import os


class MazeEnvironment:
	def __init__(self, layout, start_pos, goal_pos):
		self._grid = layout
		self._s = start_pos
		self._g = goal_pos
		

def parse_grid_from_file(file_path):
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
			elif col == 'G':
				goal = (x, y)
			elif col == '\n':
				pass
			else:
				raise ValueError('Unexpected character in maze definition file: "{}"'.format(col))
	
	assert(start != (-1, -1))
	assert(goal != (-1, -1))
	
	return MazeEnvironment(layout, start, goal)

	
if __name__ == '__main__':
	maze = parse_grid_from_file("maze.txt")

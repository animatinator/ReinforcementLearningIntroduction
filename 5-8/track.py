# The track environment and associated classes.
#
# Conventions used herein:
# * Coordinates start from (0, 0) in the top-left corner of the screen; (c, r)
#   refers to column c and row r.
# * Velocity is positive if moving up-screen (ie decreasing y coordinate) and
#   to the right (ie increasing x coordinate).
# * The track array is stored as an array of row arrays, ie coordinate (x, y)
#   maps to track[y][x].


import constants
from enum import Enum
import numpy as np
import os
from PIL import Image
import random
import sys
from utils import State, TimeStep


def read_track(filename):
	image = Image.open(os.path.join(sys.path[0], filename))
	array = np.array(image, dtype=np.float32) / 255.0
	array = array[:, :, 0]

	starts = np.where(array[image.height - 1, :] > 0)[0]
	start_range = (starts[0], starts[-1])
	ends = np.where(array[:, image.height - 1] > 0)[0]
	end_range = (ends[0], ends[-1])

	return Track(array, start_range, end_range)
	
	
class Action:
	def __init__(self, x_move, y_move):
		assert(-1 <= x_move <= 1)
		assert(-1 <= y_move <= 1)
		self.x = x_move
		self.y = y_move
	
	
class Track:
	def __init__(self, track, start_range, goal_range):
		self.track = track
		self.start_range = start_range
		self.goal_range = goal_range
	
	def crosses_goal(self, start, end):
		goal_x = len(self.track[0]) - 1
		# Must have reached the right side of the track
		if end[0] < goal_x:
			return False
			
		x_distance = float(end[0] - start[0])
		
		gradient = float(end[1] - start[1]) / x_distance
		y_at_goal = (float(goal_x - start[0]) / x_distance) * gradient
		
		return self.goal_range[0] <= int(y_at_goal) <= self.goal_range[1]
	
	def out_of_range(self, point):
		if point[0] < 0 or point[1] < 0 or point [0] >= len(self.track[0]) or point[1] >= len(self.track):
			return True
		return self.track[point[1]][point[0]] == 0


class TrackEnvironment:
	def __init__(self, track):
		self._track = track
		self.reset()
		
	def _random_start_position(self):
		y = len(self._track.track)
		x = random.randrange(self._track.start_range[0], self._track.start_range[1])
		return (x, y)
		
	def reset(self):
		self._position = self._random_start_position()
		self._velocity = (0, 0)
		
	def _clamp_velocity_to_range(self, velocity):
		if velocity < constants.MIN_VELOCITY:
			velocity = constants.MIN_VELOCITY
		if velocity > constants.MAX_VELOCITY:
			velocity = constants.MAX_VELOCITY
		
		return velocity
	
	def _compute_move(self, position, velocity):
		# We subtract 'y' velocity because the car is moving up the screen.
		# ('y' velocity is flipped to keep it positive for convenience.)
		return (position[0] + velocity[0], position[1] - velocity[1])
		
	def _get_state(self):
		return State(self._position, self._velocity)
		
	def step(self, action):
		self._velocity = (
			self._clamp_velocity_to_range(self._velocity[0] + action.x),
			self._clamp_velocity_to_range(self._velocity[1] + action.y))
		new_pos = self._compute_move(self._position, self._velocity)

		if self._track.crosses_goal(self._position, new_pos):
			return TimeStep(State(new_pos, self._velocity), constants.GOAL_REWARD)
		elif self._track.out_of_range(new_pos):
			self.reset()
			return TimeStep(self._get_state(), constants.STEP_REWARD)
		else:
			self._position = new_pos
			return TimeStep(self._get_state(), constants.STEP_REWARD)
		
class InspectableTrackEnvironent(TrackEnvironment):
	def __init__(self, track):
		super().__init__(track)
		
	def get_track(self):
		return self._track
		
	def get_state(self):
		return self._get_state()
		
	def size(self):
		return (len(self._track.track[1]), len(self._track.track))


if __name__ == '__main__':
	track = read_track("track.bmp")
	print(track.track)
	env = InspectableTrackEnvironent(track)
	action = Action(1, 1)
	print(env.step(action))
	print(env.step(action))

# The track environment and associated classes.
#
# Conventions used herein:
# * Coordinates start from (0, 0) in the top-left corner of the screen; (c, r)
#   refers to column c and row r.
# * Velocity is positive if moving up-screen (ie decreasing y coordinate) and
#   to the right (ie increasing x coordinate).
# * The track array is stored as an array of row arrays, ie coordinate (x, y)
#   maps to track[y][x].


from dataclasses import dataclass
from enum import Enum, unique
import numpy as np
import os
from PIL import Image
import random
import sys
# Truly horrendous, but couldn't find a better way of being able to run this
# both from the parent directory and this directory. Pull requests welcome.
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import constants


def read_track(filename):
	image = Image.open(os.path.join(sys.path[0], filename))
	array = np.array(image, dtype=np.float32) / 255.0
	array = array[:, :, 0]

	starts = np.where(array[image.height - 1, :] > 0)[0]
	start_range = (starts[0], starts[-1])
	ends = np.where(array[:, image.height - 1] > 0)[0]
	end_range = (ends[0], ends[-1])

	return Track(array, start_range, end_range)
	

class Action(Enum):
	NOOP = 0
	ACCEL_X = 1
	ACCEL_Y = 2
	DECEL_X = 3
	DECEL_Y = 4
	
	def get_movement(self):
		if self == Action.NOOP:
			return (0, 0)
		elif self == Action.ACCEL_X:
			return (1, 0)
		elif self == Action.ACCEL_Y:
			return (0, 1)
		elif self == Action.DECEL_X:
			return (-1, 0)
		elif self == Action.DECEL_Y:
			return (0, -1)


@dataclass
class State:
	pos: (int, int)
	vel: (int, int)


@dataclass
class TimeStep:
	state: State
	reward: float
	terminal: bool = False


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
			
		goal_pos = self.snap_to_goal(start, end)
		
		return self.goal_range[0] <= goal_pos[1] <= self.goal_range[1]
	
	def snap_to_goal(self, start, end):
		goal_x = len(self.track[0]) - 1
		x_distance = float(end[0] - start[0])
		
		if x_distance == 0:
			return (goal_x, start[1])

		gradient = float(end[1] - start[1]) / x_distance
		y_at_goal = start[1] + (float(goal_x - start[0]) / x_distance) * gradient

		return (goal_x, int(y_at_goal))
	
	def out_of_range(self, point):
		if point[0] < 0 or point[1] < 0 or point [0] >= len(self.track[0]) or point[1] >= len(self.track):
			return True
		return self.track[point[1]][point[0]] == 0
		
	def size(self):
		return (len(self.track[1]), len(self.track))


class TrackEnvironment:
	def __init__(self, track):
		self._track = track
		self.reset()
		
	def size(self):
		return self._track.size()

	def _random_start_position(self):
		y = len(self._track.track) - 1
		x = random.randrange(self._track.start_range[0], self._track.start_range[1])
		return (x, y)
		
	def reset(self):
		state = State(self._random_start_position(), (0, 0))
		return TimeStep(state, 0)
	
	def _compute_move(self, position, velocity):
		# We subtract 'y' velocity because the car is moving up the screen.
		# ('y' velocity is flipped to keep it positive for convenience.)
		return (position[0] + velocity[0], position[1] - velocity[1])

	def get_available_actions(self, state):
		actions = set()

		if state.vel[0] >= constants.MIN_VELOCITY or state.vel[1] >= constants.MIN_VELOCITY:
			actions.add(Action.NOOP)
			if state.vel[0] > 0:
				actions.add(Action.DECEL_X)
			if state.vel[1] > 0:
				actions.add(Action.DECEL_Y)
		if state.vel[0] < constants.MAX_VELOCITY:
			actions.add(Action.ACCEL_X)
		if state.vel[1] < constants.MAX_VELOCITY:
			actions.add(Action.ACCEL_Y)

		return actions
		
	def step(self, state, action):
		assert action in self.get_available_actions(state), f"Invalid action! State: {state}, action: {action}"

		movement = action.get_movement()
		velocity = (state.vel[0] + movement[0], state.vel[1] + movement[1])
		new_pos = self._compute_move(state.pos, velocity)

		if self._track.crosses_goal(state.pos, new_pos):
			new_pos = self._track.snap_to_goal(state.pos, new_pos)
			return TimeStep(State(new_pos, velocity), constants.GOAL_REWARD, terminal = True)
		elif self._track.out_of_range(new_pos):
			state = State(self._random_start_position(), (0, 0))
			return TimeStep(state, constants.STEP_REWARD)
		else:
			return TimeStep(State(new_pos, velocity), constants.STEP_REWARD)


class InspectableTrackEnvironent(TrackEnvironment):
	def __init__(self, track):
		super().__init__(track)
		
	def get_track(self):
		return self._track
		
	def get_state(self):
		return self._get_state()


if __name__ == '__main__':
	track = read_track("track.bmp")
	print(track.track)
	env = InspectableTrackEnvironent(track)
	action = Action.ACCEL_X
	print(env.step(action))
	print(env.step(action))

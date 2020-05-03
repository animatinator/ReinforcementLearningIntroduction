# Windy gridworld
#
# Conventions:
# * Co-ordinates place (0,0) in the top-left corner
# * The UP action therefore decreases the y coordinate
# * Wind applies in the UP direction, thus decreasing the y coordinate.

from dataclasses import dataclass
from enum import Enum, unique
import random


@unique
class Action(Enum):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	UPLEFT = 4
	UPRIGHT = 5
	DOWNLEFT = 6
	DOWNRIGHT = 7
	NONE = 8
	
	def get_movement(self):
		if self == Action.UP:
			return (0, -1)
		elif self == Action.DOWN:
			return (0, 1)
		elif self == Action.LEFT:
			return (-1, 0)
		elif self == Action.RIGHT:
			return (1, 0)
		elif self == Action.UPLEFT:
			return (-1, -1)
		elif self == Action.UPRIGHT:
			return (1, -1)
		elif self == Action.DOWNLEFT:
			return (-1, 1)
		elif self == Action.DOWNRIGHT:
			return (1, 1)
		elif self == Action.NONE:
			return (0, 0)
		else:
			raise AssertionError("Unsupported action type: {}".format(self))


STANDARD_ACTIONS = set([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])
KINGS_ACTIONS = STANDARD_ACTIONS.union(
	set([Action.UPLEFT, Action.UPRIGHT, Action.DOWNLEFT, Action.DOWNRIGHT]))
EXTENDED_KINGS_ACTIONS = KINGS_ACTIONS.union(set([Action.NONE]))


@dataclass
class TimeStep:
	state: (int, int)
	reward: int
	terminal: bool


class WindyGridworld:
	def __init__(self, winds, height, goal_pos, action_filter, stochastic_wind = False):
		self._winds = winds
		self._w = len(winds)
		self._h = height
		self._goal = goal_pos
		self._action_filter = action_filter
		self._stochastic_wind = stochastic_wind
		
	def available_actions(self, state):
		actions = set()
		up_poss = False
		down_poss = False
		if state[1] > 0:
			actions.add(Action.UP)
			up_poss = True
		if state[1] < self._h - 1:
			actions.add(Action.DOWN)
			down_poss = True
		if state[0] > 0:
			actions.add(Action.LEFT)
			if up_poss:
				actions.add(Action.UPLEFT)
			if down_poss:
				actions.add(Action.DOWNLEFT)
		if state[0] < self._w - 1:
			actions.add(Action.RIGHT)
			if up_poss:
				actions.add(Action.UPRIGHT)
			if down_poss:
				actions.add(Action.DOWNRIGHT)
		
		return [action for action in actions if action in self._action_filter]

	def step(self, state, action):
		assert(action in self.available_actions(state))
		(dx, dy) = action.get_movement()
		
		# Apply winds.
		x = state[0] + dx
		wind = self._winds[state[0]]  # Wind from the current state applies.
		if self._stochastic_wind:
			# Randomly vary the wind by up to one in each direction.
			wind += (random.randint(0, 2) - 1)
		dy -= wind
		y = max(0, state[1] + dy)
		y = min(self._h - 1, y)
		
		if (x, y) == self._goal:
			return TimeStep((x, y), 0, True)

		return TimeStep((x, y), -1, False)


if __name__ == '__main__':
	winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
	height = 7
	goal_pos = (7, 3)

	env = WindyGridworld(winds, height, goal_pos)

	# Check available actions work as intended.
	assert(env.available_actions((4, 4)) == set([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]))
	assert(Action.UP not in env.available_actions((4, 0)))
	assert(Action.DOWN not in env.available_actions((4, 6)))
	assert(Action.LEFT not in env.available_actions((0, 4)))
	assert(Action.RIGHT not in env.available_actions((9, 4)))

	# Check wind behaviour for a few sample cases.
	assert(env.step((9, 4), Action.UP) == TimeStep((9, 3), -1, False))
	assert(env.step((8, 4), Action.LEFT) == TimeStep(goal_pos, 0, True))
	assert(env.step((4, 0), Action.RIGHT) == TimeStep((5, 0), -1, False))
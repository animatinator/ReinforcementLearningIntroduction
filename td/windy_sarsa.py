# Windy gridworld SARSA solution
#
# Conventions:
# * Co-ordinates place (0,0) in the top-left corner
# * The UP action therefore decreases the y coordinate
# * Wind applies in the UP direction, thus decreasing the y coordinate.

from dataclasses import dataclass
from enum import Enum


class Action(Enum):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	
	
@dataclass
class TimeStep:
	state: (int, int)
	reward: int
	terminal: bool


class WindyGridworld:
	def __init__(self, winds, height, goal_pos):
		self._winds = winds
		self._w = len(winds)
		self._h = height
		self._goal = goal_pos
		
	def available_actions(self, state):
		actions = set()
		if state[1] > 0:
			actions.add(Action.UP)
		if state[1] < self._h - 1:
			actions.add(Action.DOWN)
		if state[0] > 0:
			actions.add(Action.LEFT)
		if state[0] < self._w - 1:
			actions.add(Action.RIGHT)
		
		return actions
	
	def step(self, state, action):
		assert(action in self.available_actions(state))
		dx = dy = 0
		if action == Action.UP:
			dy = -1
		elif action == Action.DOWN:
			dy = 1
		elif action == Action.LEFT:
			dx = -1
		elif action == Action.RIGHT:
			dx = 1
		
		# Apply winds
		x = state[0] + dx
		dy -= self._winds[state[0]]  # Wind from the current state applies
		y = max(0, state[1] + dy)
		
		if (x, y) == self._goal:
			return TimeStep((x, y), 0, True)

		return TimeStep((x, y), -1, False)


if __name__ == '__main__':
	winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
	height = 7
	goal_pos = (7, 3)

	env = WindyGridworld(winds, height, goal_pos)

	# Verify behaviour
	assert(Action.UP not in env.available_actions((4, 0)))
	assert(Action.DOWN not in env.available_actions((4, 6)))
	assert(Action.LEFT not in env.available_actions((0, 4)))
	assert(Action.RIGHT not in env.available_actions((9, 4)))
	assert(env.step((9, 4), Action.UP) == TimeStep((9, 3), -1, False))
	assert(env.step((8, 4), Action.LEFT) == TimeStep(goal_pos, 0, True))
	assert(env.step((4, 0), Action.RIGHT) == TimeStep((5, 0), -1, False))

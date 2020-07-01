# An extremely basic corridor environment for policy gradient experiments.

from dataclasses import dataclass
from enum import Enum, unique

@unique
class Action(Enum):
	LEFT = 0
	RIGHT = 1

	def get_movement(self):
		if self == Action.LEFT:
			return -1
		elif self == Action.RIGHT:
			return 1
		else:
			raise AssertionError(f"Unsupported action type: {self}")


@dataclass
class TimeStep:
	state: int
	reward: int
	terminal: True


# A short corridor:
# [ S ] [ -><- ] [ <-> ] [ G ]
# The agent starts on the left hand side.
# Each square has two available actions, left and right, which move in the
# corresponding directions. In the second square, however, the directions are
# reversed.
# Each transition yields a reward of -1.
# Going left in the leftmost state just transitions back to the leftmost state.
class ShortCorridor:
	def reset(self):
		return TimeStep(0, 0, False)

	def step(self, state, action):
		assert state != 3, "Cannot transition from the goal state"
		assert 0 <= state < 3, "State not in range"
		assert action in Action, "Invalid action"

		movement = action.get_movement()
		if state == 0 and movement < 0:
			movement = 0
		elif state == 1:
			movement = -movement

		new_state = state + movement

		if new_state == 3:
			return TimeStep(3, -1, True)

		return TimeStep(new_state, -1, False)


if __name__ == '__main__':
	env = ShortCorridor()

	step = env.reset()
	assert step == TimeStep(0, 0, False)
	
	step = env.step(step.state, Action.LEFT)
	assert step == TimeStep(0, -1, False)

	step = env.step(1, Action.LEFT)
	assert step == TimeStep(2, -1, False)

	step = env.step(2, Action.RIGHT)
	assert step == TimeStep(3, -1, True)

	try:
		env.step(3, Action.LEFT)
		assert False, "Should have thrown an exception when stepping from the goal state"
	except AssertionError:
		pass

	try:
		env.step(-3, Action.RIGHT)
		assert False, "Should have thrown an exception when stepping from an invalid state"
	except AssertionError:
		pass

	try:
		env.step(2, 123)
		assert False, "Should have thrown an exception when stepping with an invalid action"
	except AssertionError:
		pass

	print("Environment looks good!")

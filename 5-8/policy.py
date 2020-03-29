# Q-value functions and policies

import constants
import itertools
import numpy as np
from track import Action
from utils import State


def action_to_integer(action):
	assert(-1 <= action.x <= 1)
	assert(-1 <= action.y <= 1)
	return (action.y + 1) * 3 + (action.x + 1)


def integer_to_action(action_int):
	assert(0 <= action_int <= 8)
	return Action(action_int % 3 - 1, int(action_int / 3) - 1)


def velocity_to_integer(velocity):
	assert(constants.MIN_VELOCITY <= velocity[0] <= constants.MAX_VELOCITY)
	assert(constants.MIN_VELOCITY <= velocity[1] <= constants.MAX_VELOCITY)
	velocity_range = (constants.MAX_VELOCITY - constants.MIN_VELOCITY) + 1
	return (velocity[1]) * velocity_range + velocity[0]


class QFunction:
	def __init__(self, state_size, velocity_size):
		self.state_size = state_size
		self.velocity_size = velocity_size
		self._q = np.random.rand(state_size[1], state_size[0], velocity_size * velocity_size, 9)
		self._c = np.zeros((state_size[1], state_size[0], velocity_size * velocity_size, 9), dtype=np.float32)
		
	def get_count(self, state, action):
		return self._c[state.pos[1]][state.pos[0]][velocity_to_integer(state.vel)][action_to_integer(action)]
	
	def increment_count(self, state, action, value):
		self._c[state.pos[1]][state.pos[0]][velocity_to_integer(state.vel)][action_to_integer(action)] += value
	
	def get(self, state, action):
		return self._q[state.pos[1]][state.pos[0]][velocity_to_integer(state.vel)][action_to_integer(action)]
		
	def get_max_action(self, state):
		return np.argmax(self._q[state.pos[1]][state.pos[0]][velocity_to_integer(state.vel)])
	
	def set(self, state, action, value):
		self._q[state.pos[1]][state.pos[0]][velocity_to_integer(state.vel)][action_to_integer(action)] = value


class Policy:
	def __init__(self, state_size, velocity_size):
		self._pi = np.zeros((state_size[1], state_size[0], velocity_size * velocity_size), np.int32)
	
	def get_action(self, state):
		action_int = self._pi[state.pos[1]][state.pos[0]][velocity_to_integer(state.vel)]
		return integer_to_action(action_int)
	
	def update_with_action(self, state, action):
		self._pi[state.pos[1]][state.pos[0]][velocity_to_integer(state.vel)] = action_to_integer(action)
	
	def update(self, state, action_int):
		self._pi[state.pos[1]][state.pos[0]][velocity_to_integer(state.vel)] = action_int


def build_max_policy(q_function):
	state_size = q_function.state_size
	velocity_size = q_function.velocity_size
	policy = Policy(state_size, velocity_size)
	
	for x, y in itertools.product(range(state_size[0]), range(state_size[1])):
		for v_x, v_y in itertools.product(range(velocity_size), range(velocity_size)):
			state = State((x, y), (v_x, v_y))
			policy.update(state, q_function.get_max_action(state))
	
	return policy

	
# Simple functionality tests (because I'm too lazy to test this project properly).
if __name__ == '__main__':
	# Check that action_to_integer works as expected.
	assert(action_to_integer(Action(-1, -1)) == 0)
	assert(action_to_integer(Action(1, 1)) == 8)
	assert(action_to_integer(Action(0, 0)) == 4)
	assert(action_to_integer(Action(1, 0)) == 5)
	
	# Check integer_to_action reverses it.
	assert(integer_to_action(action_to_integer(Action(1, -1))) == Action(1, -1))
	
	# velocity_to_integer should work similarly.
	assert(velocity_to_integer((0, 0)) == 0)
	assert(velocity_to_integer((1, 0)) == 1)
	assert(velocity_to_integer((0, 1)) == 6)
	assert(velocity_to_integer((3, 3)) == 21)
	
	# Build a Q function and check we can update it.
	q_f = QFunction((70, 70), 6)
	q_f.set(State((15, 20), (2, 2)), Action(1, 1), 27)
	assert(q_f.get(State((15, 20), (2, 2)), Action(1, 1)) == 27)

	# Check the Q function can track visit counts too.
	q_f.increment_count(State((15, 20), (2, 2)), Action(1, 1), 1)
	q_f.increment_count(State((15, 20), (2, 2)), Action(1, 1), 26)
	assert(q_f.get_count(State((15, 20), (2, 2)), Action(1, 1)) == 27)
	
	# A maximising policy should now choose the action we assigned the value of
	# 27 whenever we're in that state.
	policy = build_max_policy(q_f)
	assert(policy.get_action(State((15, 20), (2, 2))) == Action(1, 1))
	
	print("All smoke tests passed.")

# Very basic problem solved with temporal-difference learning.
# The setup (X terminal, S normal state):
# [X S S S S S S X]
# Transitions left and right give a reward of zero, apart from a transition to
# the rightmost terminal which gives a reward of one.

from dataclasses import dataclass
from enum import Enum
import random


# The number of non-terminal states
STATES = 5
# The state to start in (zero-based index, including left terminal).
START_STATE = 3

ALPHA = 0.1
TRAIN_STEPS = 100


@dataclass
class TimeStep:
	state: int
	reward: int
	terminal: bool
	
class Action(Enum):
	LEFT = 0
	RIGHT = 1


def is_terminal(state):
	return state == 0 or state == (STATES + 1)


def is_goal(state):
	return state == (STATES + 1)
	

def make_step(state, action):
	if action == Action.LEFT:
		new_state = state - 1
	else:
		new_state = state + 1
	
	reward = 1 if is_goal(new_state) else 0
	return TimeStep(new_state, reward, is_terminal(new_state))
	

def get_value(state, values):
	if is_terminal(state):
		return 0
	return values[state-1]


def update_value(state, values, value):
	if not is_terminal(state):
		values[state-1] = value


def run_td(values):
	state = START_STATE
	step = TimeStep(START_STATE, 0, False)
	
	while not step.terminal:
		action = Action(random.randint(0, len(Action) - 1))
		step = make_step(state, action)

		old_val = get_value(state, values)
		next_state_val = get_value(step.state, values)
		new_val = old_val + ALPHA * ((step.reward + next_state_val) - old_val)
		update_value(state, values, new_val)
		
		state = step.state


if __name__ == '__main__':
	values = [0.5] * 5

	for i in range(0, TRAIN_STEPS):
		run_td(values)
	
	print(values)
# Very basic problem solved with temporal-difference learning.
# The setup (X terminal, S normal state):
# [X S S S S S S X]
# Transitions left and right give a reward of zero, apart from a transition to
# the rightmost terminal which gives a reward of one.

from dataclasses import dataclass
from enum import Enum
import math
from matplotlib import pyplot
import random


# The number of non-terminal states
STATES = 15
# The state to start in (zero-based index, including left terminal).
START_STATE = 3

ALPHA = 0.05
TRAIN_STEPS = 10000
N_STEP_ALPHA = 0.4
N = 4


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


def rms_error(values):
	sum = 0.0
	for i in range(len(values)):
		expected_value = float(i + 1) / float(len(values) + 1)
		sum += (values[i] - expected_value) ** 2
	return math.sqrt(sum)


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


# fix_values is as specified in the comment on run_n_step_td.
def n_step_update(past_results, values, fix_values):
	state_to_update = past_results[0][0]
	# Potentially use old values.
	read_values = past_results[0][2] if fix_values else values
	latest_state = past_results[-1][0]
	past_results.pop(0)
	latest_estimate = get_value(latest_state, read_values)
	reward_sum = sum([entry[1] for entry in past_results]) + latest_estimate
	old_val = get_value(state_to_update, values)
	new_val = old_val + ALPHA * (reward_sum - old_val)
	update_value(state_to_update, values, new_val)


# fix_values specifies that an update to a state should use the state values
# from the time the state was reached (ignoring subsequent updates). This is
# less accurate, but is exercise 7.2 in Reinforcement Learning: An Introduction.
def run_n_step_td(values, n, fix_values=False):
	state = START_STATE
	step = TimeStep(START_STATE, 0, False)

	past_results = [(START_STATE, 0, values.copy())]

	while not step.terminal:
		action = Action(random.randint(0, len(Action) - 1))
		step = make_step(state, action)
		past_results.append((step.state, step.reward, values.copy()))

		if len(past_results) > n + 1:
			n_step_update(past_results, values, fix_values)

		state = step.state

	while past_results:
		n_step_update(past_results, values, fix_values)


def moving_average(values, n):
	results = [0] * len(values)
	for i in range(1, len(values)):
		real_n = min(i, n)
		moving_val = float(sum(values[i-real_n:i])) / float(real_n)
		results[i] = moving_val

	return results


def plot_errors(errors):
	ys = moving_average(errors, 30)
	xs = [i for i in range(len(errors))]
	pyplot.scatter(xs, ys)
	pyplot.show()


def run_and_evaluate_td_method(method, name):
	values = [0.5] * STATES
	errors = []

	for i in range(0, TRAIN_STEPS):
		method(values)
		errors.append(rms_error(values))

	print("{} result: {}".format(name, values))
	print("Drawing error graph...")
	plot_errors(errors)


if __name__ == '__main__':
	run_and_evaluate_td_method(run_td, "Simple TD")
	run_and_evaluate_td_method(lambda values: run_n_step_td(values, N), "{}-step TD".format(N))
	run_and_evaluate_td_method(lambda values: run_n_step_td(values, N, True), "{}-step TD".format(N))

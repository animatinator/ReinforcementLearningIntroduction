# Calculating probabilities of success using value iteration

from matplotlib import pyplot
import numpy as np

TOP_STATE = 100
ALLOWED_ERROR = 0.0001

def _value_of_action(value_fn, state, stake, p_heads):
	assert(0 < state < 100)
	return (p_heads * value_fn[state + stake]) + ((1.0 - p_heads) * value_fn[state - stake])
	
def _get_updated_value(value_fn, state, p_heads):
	if state == 0:
		return 0
	if state == 100:
		return 1

	max_val = 0

	for possible_stake in range(0, min(state, TOP_STATE - state) + 1):
		max_val = max(max_val, _value_of_action(value_fn, state, possible_stake, p_heads))
	
	return max_val

def _value_iterate(p_heads):
	max_diff = 100.0
	
	value_fn = [0]*TOP_STATE
	value_fn.append(1)
	
	while max_diff > ALLOWED_ERROR:
		max_diff = 0.0
		for state in range(1, TOP_STATE):
			new_val = _get_updated_value(value_fn, state, p_heads)
			max_diff = max(max_diff, abs(new_val - value_fn[state]))
			value_fn[state] = new_val

	return value_fn

def _compute_policy(value_fn, p_heads):
	policy = [0] * 99
	
	for state in range(1, TOP_STATE):
		max_policy = 0
		max_val = 0
		for possible_stake in range(0, min(state, TOP_STATE - state) + 1):
			val_here = _value_of_action(value_fn, state, possible_stake, p_heads)
			if val_here > max_val:
				max_val = val_here
				max_policy = possible_stake
		policy[state - 1] = max_policy
	
	return policy

def _evaluate_and_visualise(p_heads):
	value_fn = _value_iterate(p_heads)
	pyplot.plot(range(0, TOP_STATE + 1), value_fn)
	pyplot.show()

	policy = _compute_policy(value_fn, p_heads)
	pyplot.plot(range(1, TOP_STATE), policy)
	pyplot.show()

if __name__ == '__main__':
	_evaluate_and_visualise(0.25)
	_evaluate_and_visualise(0.55)

# On-policy first-visit MC control with a soft policy

import environment
import numpy as np
import policy


def rollout(policy):
	episode = []

	env = environment.Blackjack()
	timestep = env.reset()
	
	while not timestep.terminal:
		state = timestep.observation
		action = policy.act(state[0], state[1], state[2])
		old_step = timestep
		timestep = env.step(action)
		episode.append((old_step.observation, action, timestep.reward))
	
	return episode


if __name__ == '__main__':
	policy = policy.Policy()
	print(rollout(policy))

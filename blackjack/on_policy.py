# On-policy first-visit MC control with a soft policy

import environment
import numpy as np


def rollout(policy):
	episode = []

	env = environment.Blackjack()
	timestep = env.reset()
	
	while not timestep.terminal:
		state = timestep.observation
		action = policy[state[0]][state[1]][1 if state[2] else 0]
		old_step = timestep
		print(policy[state[0]][state[1]])
		timestep = env.step(action)
		episode.append((old_step.observation, action, timestep.reward))
	
	return episode


if __name__ == '__main__':
	policy = np.ones((22, 11, 2), dtype=np.bool)
	print(policy)
	print(rollout(policy))

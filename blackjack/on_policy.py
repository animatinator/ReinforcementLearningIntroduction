# On-policy first-visit MC control with a soft policy

import environment
import matplotlib.pyplot as plt
import numpy as np
import policy


EPSILON = 0.1
LAMBDA = 0.9
TRAIN_STEPS = 20000
REPORT_EVERY = 100


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


def process_for_first_visits(rollout):
	seen = set()
	result = []

	for step in rollout:
		if (step[0], step[1]) in seen:
			result.append(step + [False])
		else:
			seen.add((step[0], step[1]))
			result.append(step + (True,))

	return result
	
	
def learn_from_episode(episode, pi, q, returns):
	episode = process_for_first_visits(episode)
	G = 0

	for i in range(len(episode) - 1, -1, -1):
		step = episode[i]
		G = (LAMBDA * G) + step[2]

		if step[3]:  # Indicates whether this is the first visit.
			observation = step[0]
			player_score = observation[0]
			dealer_score = observation[1]
			usable_ace = observation[2]
			action = step[1]
			returns.add_return(player_score, dealer_score, usable_ace, action, G)
			q.update(player_score, dealer_score, usable_ace, action,
				returns.average_return(player_score, dealer_score, usable_ace, action))
			best_action = q.pick_best(player_score, dealer_score, usable_ace)

			policy_update = policy.e_soften(
				np.array([0.0, 1.0]) if best_action else np.array([1.0, 0.0]), EPSILON)
			pi.set(player_score, dealer_score, usable_ace, policy_update)
	

def train_policy():
	pi = policy.Policy()
	# Stick on 20 or 21.
	pi.set(20, 0, False, np.array([1.0, 0.0]))
	pi.set(21, 0, False, np.array([1.0, 0.0]))
	pi.set(20, 0, True, np.array([1.0, 0.0]))
	pi.set(21, 0, True, np.array([1.0, 0.0]))
	
	pi.e_soften(EPSILON)
	q = policy.QValues()
	returns = policy.ReturnCounter()

	for i in range(TRAIN_STEPS):
		if i % REPORT_EVERY == 0:
			print("Train step {}...".format(i))
		episode = rollout(pi)
		learn_from_episode(episode, pi, q, returns)

	return pi


def plot_policy(policy):
	p = np.random.randint(2, size=(100, 100))
	no_ace = policy._pi[:, :, 0, 1]
	ace = policy._pi[:, :, 1, 1]
	plt.imshow(no_ace)
	plt.show()
	plt.imshow(ace)
	plt.show()


if __name__ == '__main__':
	policy = train_policy()
	plot_policy(policy)

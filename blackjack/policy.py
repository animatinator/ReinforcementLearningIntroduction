# Policy representations

import numpy as np
from numpy import random


def e_soften(array, e):
	softened = array * (1.0 - e) + (e / len(array))
	# Normalise before returning.
	return softened / float(sum(softened))


class ReturnCounter:
	def __init__(self):
		self._r = [[[[[]] * 2] * 2] * 11] * 22

	def add_return(self, player_score, dealer_score, usable_ace, action, r):
		self._r[player_score][dealer_score][usable_ace][action].append(r)

	def average_return(self, player_score, dealer_score, usable_ace, action):
		return np.average(self._r[player_score][dealer_score][usable_ace][action])


class QValues:
	def __init__(self):
		# (player_score, dealer_score, usable_ace)
		self._q = random.rand(22, 11, 2)
		
	def get(self, player_score, dealer_score, usable_ace):
		return self._q[player_score][dealer_score][usable_ace]
	
	def update(self, player_score, dealer_score, usable_ace, new_value):
		self._q[player_score][dealer_score][usable_ace] = new_value


class Policy:
	def __init__(self):
		# (player_score, dealer_score, usable_ace, stick/hit)
		self._pi = np.ones((22, 11, 2, 2))
		self._pi[:, :, :, 0] = False
		
	def e_soften(self, epsilon):
		for p in range(22):
			for d in range(11):
				for u in range(2):
					self._pi[p][d][u] = e_soften(self._pi[p][d][u], epsilon)
	
	def act(self, player_score, dealer_score, usable_ace):
		action_probabilities = self._pi[player_score][dealer_score][1 if usable_ace else 0]
		return random.choice([0, 1], p=action_probabilities) == 1

		
if __name__ == '__main__':
	policy = Policy()
	assert(policy.act(10, 0, True) == True)

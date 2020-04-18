# Policy representations

import numpy as np
from numpy import random


def bool_to_int(bool_val):
	return 1 if bool_val else 0


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
		# (player_score, dealer_score, usable_ace, stick/hit)
		self._q = random.rand(22, 11, 2, 2)
		
	def get(self, player_score, dealer_score, usable_ace, action):
		action = bool_to_int(action)
		usable_ace = bool_to_int(usable_ace)
		return self._q[player_score][dealer_score][usable_ace][action]
	
	def update(self, player_score, dealer_score, usable_ace, action, new_value):
		action = bool_to_int(action)
		usable_ace = bool_to_int(usable_ace)
		self._q[player_score][dealer_score][usable_ace][action] = new_value

	def pick_best(self, player_score, dealer_score, usable_ace):
		usable_ace = bool_to_int(usable_ace)
		possibilities = self._q[player_score][dealer_score][usable_ace]
		return np.argmax(possibilities)


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

	def set(self, player_score, dealer_score, usable_ace, values):
		usable_ace = bool_to_int(usable_ace)
		self._pi[player_score][dealer_score][usable_ace] = values
	
	def act(self, player_score, dealer_score, usable_ace):
		action_probabilities = self._pi[player_score][dealer_score][1 if usable_ace else 0]
		return random.choice([0, 1], p=action_probabilities) == 1

		
if __name__ == '__main__':
	policy = Policy()
	assert(policy.act(10, 0, True) == True)

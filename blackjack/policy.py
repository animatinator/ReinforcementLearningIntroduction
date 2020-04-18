# Policy representations

import numpy as np
from numpy import random

class Policy:
	def __init__(self):
		# (player_score, dealer_score, usable_ace, stick/hit)
		self._pi = np.ones((22, 11, 2, 2), dtype=np.bool)
		self._pi[:, :, :, 0] = False
	
	def act(self, player_score, dealer_score, usable_ace):
		action_probabilities = self._pi[player_score][dealer_score][1 if usable_ace else 0]
		return random.choice([0, 1], p=action_probabilities) == 1

		
if __name__ == '__main__':
	policy = Policy()
	assert(policy.act(10, 0, True) == True)

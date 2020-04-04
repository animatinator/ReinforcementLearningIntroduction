# Utilities for working with episode data.

class EpisodeStep:
	def __init__(self, state, action, reward):
		self.state = state
		self.action = action
		self.reward = reward
		
	def __str__(self):
		return "Episode step: Reward {}; State {} -> {}".format(
			self.reward, self.state, self.action)
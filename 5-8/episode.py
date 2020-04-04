# Utilities for working with episode data.

class EpisodeStep:
	def __init__(self, state, action, reward):
		self.state = state
		self.action = action
		self.reward = reward
		
	def __str__(self):
		return "Episode step: Reward {}; State {} -> {}".format(
			self.reward, self.state, self.action)
			

class Episode:
	def __init__(self):
		self._steps = []
		self._reward = 0
		
	def get_steps(self):
		return self._steps
		
	def get_total_reward(self):
		return self._reward
	
	def add_step(self, step):
		self._steps.append(step)
		self._reward += step.reward
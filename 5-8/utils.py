# Utility classes and methods.
class State:
	def __init__(self, position, velocity):
		self.pos = position
		self.vel = velocity
		
	def __str__(self):
		return "pos: {}, vel: {}".format(self.pos, self.vel)


class TimeStep:
	def __init__(self, state, reward):
		self.state = state
		self.reward = reward
	
	def __str__(self):
		return "State: [{}], reward: {}".format(self.state, self.reward)
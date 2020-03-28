# Utility classes and methods.
class State:
	def __init__(self, position, velocity):
		self.pos = position
		self.vel = velocity
		
	def __str__(self):
		return "pos: {}, vel: {}".format(self.pos, self.vel)


class TimeStep:
	def __init__(self, state, reward, terminal = False):
		self.state = state
		self.reward = reward
		self.terminal = terminal
	
	def __str__(self):
		return "{}: [{}], reward: {}".format(
			"Terminal state" if self.terminal else "State",
			self.state,
			self.reward)
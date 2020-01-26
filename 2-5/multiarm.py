# Multi-armed bandit experiments

from matplotlib import pyplot
import numpy as np
from numpy import random
import random as py_rand

START_VAL = 1.0
WALK_STDDEV = 0.3
NUM_AGENTS = 30
NUM_STEPS = 30000

def get_maximum(array):
	max = -1000000.0
	maxes = []
	
	for i, elem in enumerate(array):
		if elem > max:
			maxes = [i]
			max = elem
		elif elem == max:
			maxes.append(i)
		
	return py_rand.choice(maxes)

def plot_average_reward(agents, colour):
	numSteps = len(agents[0].rewardHistory)
	steps = range(numSteps)
	averages = [np.average([agent.rewardHistory[s] for agent in agents]) for s in steps]
	pyplot.plot(steps, averages, color=colour)

class BanditArm(object):
	def __init__(self, initial, randomWalk=False):
		self.mean = initial
		self.randomWalk = randomWalk
	
	def step(self):
		reward = random.normal(self.mean)
		if self.randomWalk:
			self.mean += random.normal(0, WALK_STDDEV)
		return reward

		
class MultiArmEnvironment(object):
	def __init__(self, num_arms):
		self._init_arms(num_arms)
		
	def _init_arms(self, num_arms):
		self.arms = [BanditArm(1.0, False) for i in range(num_arms)]
		
	def step(self, action):
		reward = self.arms[action].step()
		return reward
		
	def arm_count(self):
		return len(self.arms)
		

class WalkingMultiArmEnvironment(MultiArmEnvironment):
	def _init_arms(self, num_arms):
		self.arms = [BanditArm(1.0, True) for i in range(num_arms)]


class PlotterAgent(object):
	def __init__(self, env):
		self.env = env
		
	def plot(self, numTests):
		numArms = self.env.arm_count()
		xs = np.repeat(range(numArms), numTests)
		ys = [self.env.step(n) for n in xs]
		
		pyplot.scatter(xs, ys)
		pyplot.show()
		
		
class SampleAverageAgent(object):
	def __init__(self, env, eGreedy):
		self.env = env
		self.q = [0.0 for _ in range(self.env.arm_count())]
		self.counts = [0 for _ in range(self.env.arm_count())]
		self.eGreedy = eGreedy
		self.rewardHistory = []
		
	def step(self):
		if random.uniform() > self.eGreedy:
			index = get_maximum(self.q)
		else:
			index = random.randint(len(self.q))
			
		reward = self.env.step(index)
		self.rewardHistory.append(reward)
		
		self.counts[index] += 1
		self.q[index] += (reward - self.q[index]) / self.counts[index]
		
		
class NonStationaryAgent(object):
	def __init__(self, env, eGreedy):
		self.env = env
		self.q = [0.0 for _ in range(self.env.arm_count())]
		self.counts = [0 for _ in range(self.env.arm_count())]
		self.eGreedy = eGreedy
		self.rewardHistory = []
		self.stepSize = 0.1
		
	def step(self):
		if random.uniform() > self.eGreedy:
			index = get_maximum(self.q)
		else:
			index = random.randint(len(self.q))
			
		reward = self.env.step(index)
		self.rewardHistory.append(reward)
		
		self.counts[index] += 1
		self.q[index] += (reward - self.q[index]) * self.stepSize
		
		
def generate_and_run_agents(numAgents, constructor):
	agents = []
	
	for a in range(NUM_AGENTS):
		bandits = WalkingMultiArmEnvironment(10)
		
		agent = constructor(bandits, 0.1)
		agents.append(agent)
		
		for i in range(NUM_STEPS):
			agent.step()
			
	return agents


if __name__ == '__main__':
	#bandits = WalkingMultiArmEnvironment(10)
	#plotAgent = PlotterAgent(bandits)
	#plotAgent.plot(25)

	sa_agents = generate_and_run_agents(NUM_AGENTS, SampleAverageAgent)
	ns_agents = generate_and_run_agents(NUM_AGENTS, NonStationaryAgent)

	plot_average_reward(sa_agents, 'green')
	plot_average_reward(ns_agents, 'blue')
	pyplot.show()
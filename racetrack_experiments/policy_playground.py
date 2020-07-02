# A wrapper around racetrack/track_playgound.py that passes in a policy controller.

import argparse
import pdb
import pickle
from racetrack import track_playground
from sarsa_lambda import *
from save_policy import POLICY_FILE


class PolicyController:
	def __init__(self, env, policy):
		self._env = env
		self._policy = policy
		self.reset()
	
	def reset(self):
		self._state = self._env.reset().state
		self._total_reward = 0
	
	def step(self, events):
		pdb.set_trace()
		action = self._policy.get_action(self._state, self._env.get_available_actions(self._state))
		timestep = env.step(self._state, action)
		self._state = timestep.state
		self._total_reward += timestep.reward
		print(timestep)
		
		if (timestep.terminal):
			print("Goal reached! Total reward: {}".format(self._total_reward))
			print("Resetting")
			self._state = env.reset().state
			self._total_reward = 0

	def get_state(self):
		return self._state


def load_policy(name=POLICY_FILE):
	serialised_dict = pickle.Unpickler(open(name, 'rb')).load()
	tiling = deserialise_coarse_tiling(serialised_dict)
	return ApproximatedPolicy(tiling)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--control')
	args = parser.parse_args()
	
	policy = load_policy()
	controller_factory = lambda env: PolicyController(env, policy)
	
	track_playground.main(controller_factory)

# Utilities for saving and loading policies

import pickle


POLICY_FILE = 'saved_policy'


def save_policy(policy):
	file = open(POLICY_FILE, 'wb')
	pickle.Pickler(file).dump(policy)


def load_policy():
	return pickle.Unpickler(open(POLICY_FILE, 'rb')).load()

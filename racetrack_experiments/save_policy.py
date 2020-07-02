# Utilities for saving and loading policies

import pickle


POLICY_FILE = 'saved_policy'


def save_policy(policy, name=POLICY_FILE):
	file = open(name, 'wb')
	pickle.Pickler(file).dump(policy)

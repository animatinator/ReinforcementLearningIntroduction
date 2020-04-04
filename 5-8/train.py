from constants import *
from episode import EpisodeStep
from policy import build_max_policy, EpsilonGreedyPolicy, Policy, QFunction
from track import Action, TrackEnvironment, read_track


def rollout(env, policy):
	state = env.reset().state
	reward = 0
	terminal = False
	episode = []

	while not terminal:
		action = policy.get_action(state)
		episode.append(EpisodeStep(state, action, reward))

		timestep = env.step(action)
		state = timestep.state
		reward = timestep.reward
		terminal = timestep.terminal
	
	# Append the goal state and final reward (no action to report here).
	episode.append(EpisodeStep(state, None, reward))
	
	return episode


if __name__ == '__main__':
	track = read_track('track.bmp')
	env = TrackEnvironment(track)

	Q = QFunction(track.size(), VELOCITY_RANGE)
	Pi = build_max_policy(Q)
	
	for i in range(TRAIN_STEPS):
		if i % REPORT_EVERY == 0:
			print("Training step {}...".format(i))
		soft_policy = EpsilonGreedyPolicy(EPSILON, Pi, NUM_ACTIONS)
		episode = rollout(env, soft_policy)
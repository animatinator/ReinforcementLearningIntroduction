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
		soft_policy = EpsilonGreedyPolicy(EPSILON, Pi, NUM_ACTIONS)
		episode = rollout(env, soft_policy)
		
		
		if i % REPORT_EVERY == 0:
			print("Training step {}...".format(i))
			print("Episode length: {}".format(len(episode)))
		
		T = len(episode) - 1
		
		G = 0.0
		W = 1.0
		
		for t in range(T-1, -1, -1):
			# Get key variables from this episode step.
			St = episode[t].state
			At = episode[t].action
			Rt_1 = episode[t+1].reward

			# Update Q-values and visit counts.
			G = (LAMBDA * G) + Rt_1
			Q.increment_count(St, At, W)
			Qs_a = Q.get(St, At)
			new_Qs_a = Qs_a + (W / Q.get_count(St, At)) * (G - Qs_a)
			Q.set(St, At, new_Qs_a)
			
			# Update the policy.
			Pi.update(St, Q.get_max_action(St))
			
			# Stop this episode if it's no longer behaving greedily.
			if At != Pi.get_action(St):
				break
			
			W += 1.0 / soft_policy.action_probability(St, At)
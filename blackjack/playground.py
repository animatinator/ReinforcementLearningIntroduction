import environment

def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
          score, usable_ace, dealer_score))

def strategy(observation):
    score, dealer_score, usable_ace = observation
    # Stick (action 0) if the score is > 20, hit (action 1) otherwise
    return 0 if score >= 20 else 1

	
if __name__ == '__main__':
	env = environment.Blackjack()
	for i_episode in range(20):
		timestep = env.reset()
		observation = timestep.observation
		for t in range(100):
			print_observation(observation)
			action = strategy(observation)
			print("Taking action: {}".format( ["Stick", "Hit"][action]))
			timestep = env.step(action)
			observation = timestep.observation
			if timestep.terminal:
				print_observation(observation)
				print("Game end. Reward: {}\n".format(float(timestep.reward)))
				break

# Windy gridworld SARSA solution

from windy_env import Action, TimeStep, WindyGridworld


if __name__ == '__main__':
	winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
	height = 7
	goal_pos = (7, 3)

	env = WindyGridworld(winds, height, goal_pos)

# A PyGame tool for testing out the track environment.

import argparse
from policy import Policy
import pygame
from track import Action, InspectableTrackEnvironent, read_track


TILE_SIZE = 6

GRASS_COLOUR = (100, 200, 100)
ROAD_COLOUR = (50, 50, 50)
CAR_COLOUR = (200, 200, 0)


class HumanController:
	def __init__(self, env):
		self._env = env
		self._total_reward = 0
		
	def step(self, events):
		for event in events:
			if event.type == pygame.KEYDOWN:
				if event.key not in {pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN, pygame.K_SPACE}:
					continue
				
				action = Action(0, 0)
				if event.key == pygame.K_LEFT:
					action = Action(-1, 0)
				if event.key == pygame.K_RIGHT:
					action = Action(1, 0)
				if event.key == pygame.K_UP:
					action = Action(0, 1)
				if event.key == pygame.K_DOWN:
					action = Action(0, -1)
				
				timestep = env.step(action)
				self._total_reward += timestep.reward
				print(timestep)
				
				if (timestep.terminal):
					print("Goal reached! Total reward: {}".format(self._total_reward))
					print("Resetting")
					env.reset()
					self._total_reward = 0


class PolicyController:
	def __init__(self, env, policy):
		self._env = env
		self._policy = policy
		self._total_reward = 0
		self._state = env.reset().state
	
	def step(self, events):
		action = self._policy.get_action(self._state)
		timestep = env.step(action)
		self._state = timestep.state
		self._total_reward += timestep.reward
		print(timestep)
		
		if (timestep.terminal):
			print("Goal reached! Total reward: {}".format(self._total_reward))
			print("Resetting")
			self._state = env.reset().state
			self._total_reward = 0


def _draw_env(env, screen):
	screen.fill(GRASS_COLOUR)

	track = env.get_track().track
	
	for y in range(0, len(track)):
		for x in range(0, len(track[y])):
			if track[y][x]:
				pygame.draw.rect(screen, ROAD_COLOUR, (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
	
	pos = env.get_state().pos
	pygame.draw.rect(screen, CAR_COLOUR, (pos[0] * TILE_SIZE, pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE))


def game_loop(env, controller):
	pygame.display.set_caption("Track environment playground")
	size = env.size()
	screen = pygame.display.set_mode([size[0] * TILE_SIZE, size[1] * TILE_SIZE])
	
	clock = pygame.time.Clock()

	running = True
	while running:
		events = pygame.event.get()
		for event in events:
			if event.type == pygame.QUIT:
				running = False
		
		controller.step(events)
		
		_draw_env(env, screen)
		pygame.display.flip()
		
		clock.tick(10)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--control')
	args = parser.parse_args()
	
	pygame.init()

	track = read_track('track.bmp')
	env = InspectableTrackEnvironent(track)
	
	if args.control == 'human':
		controller = HumanController(env)
	elif args.control == 'policy':
		policy = Policy(env.size(), 6)
		controller = PolicyController(env, policy)
	else:
		raise ValueError('Invalid control type: {}. Try \'human\' or \'policy\'.'.format(args.control))
	
	game_loop(env, controller)
	
	pygame.quit()

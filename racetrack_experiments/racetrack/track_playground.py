# A PyGame tool for testing out the track environment.

import os
import pygame
from .track import Action, InspectableTrackEnvironent, read_track


TILE_SIZE = 6

GRASS_COLOUR = (100, 200, 100)
ROAD_COLOUR = (50, 50, 50)
CAR_COLOUR = (200, 200, 0)


class HumanController:
	def __init__(self, env):
		self._env = env
		self.reset()
		
	def reset(self):
		self._state = self._env.reset().state
		self._total_reward = 0

	def step(self, events):
		for event in events:
			if event.type == pygame.KEYDOWN:
				if event.key not in {pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN, pygame.K_SPACE}:
					continue
				
				action = Action.NOOP
				if event.key == pygame.K_LEFT:
					action = Action.DECEL_X
				if event.key == pygame.K_RIGHT:
					action = Action.ACCEL_X
				if event.key == pygame.K_UP:
					action = Action.ACCEL_Y
				if event.key == pygame.K_DOWN:
					action = Action.DECEL_Y
				
				timestep = self._env.step(self._state, action)
				self._total_reward += timestep.reward
				self._state = timestep.state
				print(timestep)
				
				if (timestep.terminal):
					print("Goal reached! Total reward: {}".format(self._total_reward))
					print("Resetting")
					self.reset()
					self._total_reward = 0

	def get_state(self):
		return self._state


def _draw_env(env, screen, state):
	screen.fill(GRASS_COLOUR)

	track = env.get_track().track
	
	for y in range(0, len(track)):
		for x in range(0, len(track[y])):
			if track[y][x]:
				pygame.draw.rect(screen, ROAD_COLOUR, (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
	
	pos = state.pos
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
			elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
				controller.reset()
		
		controller.step(events)
		
		_draw_env(env, screen, controller.get_state())
		pygame.display.flip()
		
		clock.tick(10)


def main(controller_factory):
	pygame.init()

	path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'track.bmp')
	track = read_track(path)
	env = InspectableTrackEnvironent(track)
	controller = controller_factory(env)

	game_loop(env, controller)

	pygame.quit()


if __name__ == '__main__':
	main(lambda env: HumanController(env))

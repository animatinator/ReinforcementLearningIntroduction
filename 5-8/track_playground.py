# A PyGame tool for testing out the track environment.

import pygame
from track import Action, InspectableTrackEnvironent, read_track


TILE_SIZE = 6

GRASS_COLOUR = (100, 200, 100)
ROAD_COLOUR = (50, 50, 50)
CAR_COLOUR = (200, 200, 0)


def _draw_env(env, screen):
	screen.fill(GRASS_COLOUR)

	track = env.get_track().track
	
	for y in range(0, len(track)):
		for x in range(0, len(track[y])):
			if track[y][x]:
				pygame.draw.rect(screen, ROAD_COLOUR, (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
	
	pos = env.get_state().pos
	pygame.draw.rect(screen, CAR_COLOUR, (pos[0] * TILE_SIZE, pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE))


def game_loop(env):
	size = env.size()
	screen = pygame.display.set_mode([size[0] * TILE_SIZE, size[1] * TILE_SIZE])

	running = True
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_LEFT:
					print(env.step(Action(-1, 0)))
				if event.key == pygame.K_RIGHT:
					print(env.step(Action(1, 0)))
				if event.key == pygame.K_UP:
					print(env.step(Action(0, 1)))
				if event.key == pygame.K_DOWN:
					print(env.step(Action(0, -1)))
				if event.key == pygame.K_SPACE:
					env.step(Action(0, 0))
		
		_draw_env(env, screen)
		pygame.display.flip()


if __name__ == '__main__':
	pygame.init()

	track = read_track("track.bmp")
	env = InspectableTrackEnvironent(track)
	
	game_loop(env)
	
	pygame.quit()

import pygame

import config


class Renderer:
    def __init__(self, arena):
        pygame.init()
        self.screen = pygame.display.set_mode(config.arena_size)
        self.clock = pygame.time.Clock()
        self.running = True

        self.arena = arena

    @property
    def dt(self):
        return self.clock.tick(60) / 1000

    def get_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        keys = pygame.key.get_pressed()

        return {
            "left": keys[pygame.K_a],
            "right": keys[pygame.K_d],
            "jump": keys[pygame.K_SPACE],
        }

    def draw(self, fighters):
        self.screen.fill((30, 30, 30))
        for platform in self.arena.platforms:
            pygame.draw.rect(
                self.screen,
                (120, 120, 120),
                (round(platform.x), round(platform.y), platform.width, platform.height),
            )

        for fighter in fighters:
            pygame.draw.rect(
                self.screen,
                fighter.color,
                (round(fighter.x), round(fighter.y), fighter.width, fighter.height),
            )
        pygame.display.flip()

    def close(self):
        self.running = False
        pygame.quit()

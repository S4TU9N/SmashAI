import pygame
from pygame.freetype import Font as _Font
from pygame.freetype import get_default_resolution, init

import config


class Renderer:
    def __init__(self, arena):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(config.arena_size)
        self.clock = pygame.time.Clock()
        self.running = True

        self.arena = arena

        self.font = pygame.font.Font(None, 48)

    @property
    def dt(self):
        return self.clock.tick(60) / 1000

    def get_input(self):

        keys = pygame.key.get_pressed()

        return {
            "left": keys[pygame.K_a],
            "right": keys[pygame.K_d],
            "jump": keys[pygame.K_SPACE],
            "jab": keys[pygame.K_j],
            "heavy": keys[pygame.K_k],
        }

    def poll_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def draw(self, fighters, match):
        self.screen.fill((30, 30, 30))
        for platform in self.arena.platforms:
            pygame.draw.rect(
                self.screen,
                (120, 120, 120),
                (round(platform.x), round(platform.y), platform.width, platform.height),
            )

        for fighter in fighters:
            hitbox = fighter.attack_hitbox
            if hitbox:
                pygame.draw.rect(self.screen, (0, 255, 0), hitbox, 2)
            if fighter not in match.eliminated:
                pygame.draw.rect(
                    self.screen,
                    fighter.color,
                    (round(fighter.x), round(fighter.y), fighter.width, fighter.height),
                )
        self.draw_hud(match)
        pygame.display.flip()

    def draw_hud(self, match):
        fighters = sorted(
            match.fighters,
            key=lambda fighter: match.get_spawn_position(fighter)[0],
        )

        for index, fighter in enumerate(fighters):
            screen_width, screen_height = self.screen.get_size()

            positions = [
                (screen_width * 0.25, screen_height - 70),
                (screen_width * 0.75, screen_height - 70),
            ]
            x, y = positions[index]

            damage = f"{fighter.damage_percent:.0f}%"
            text = self.font.render(damage, True, fighter.color)

            self.screen.blit(text, (x, y))

    def close(self):
        self.running = False
        pygame.quit()

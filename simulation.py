import random as r

import config


class Platform:
    def __init__(self, x, y, width, height=20):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.width

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.height


class Fighter:
    def __init__(
        self,
        speed=600,
        gravity=2000,
        jump_height=1000,
        x=0,
        y=0,
        width=50,
        height=80,
    ):
        self.dx = 0.0
        self.dy = 0.0

        self.x = x * config.ARENA_SCALE
        self.y = y * config.ARENA_SCALE

        self.width = width
        self.height = height

        self.speed = speed
        self.gravity = gravity
        self.jump_height = jump_height

        self.grounded = False

        self.color = (
            int(r.random() * 255),
            int(r.random() * 255),
            int(r.random() * 255),
        )

    def can_jump(self) -> bool:
        return self.grounded

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.width

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.height

    def tick(self, dt, platforms):
        previous_bottom = self.bottom
        detect_collide = False
        self.dy += self.gravity * dt

        self.x += self.dx * dt
        self.y += self.dy * dt

        for platform in platforms:
            collision_detected = (
                self.dy > 0
                and previous_bottom <= platform.top
                and self.bottom >= platform.top
                and self.right > platform.left
                and self.left < platform.right
            )

            if collision_detected:
                self.y = platform.top - self.height
                self.dy = 0
                detect_collide = True
                break

        self.grounded = detect_collide

    def apply_input(self, actions):
        move = 0
        jump = False

        for action in actions:
            match action:
                case "move_left":
                    move -= 1
                case "move_right":
                    move += 1
                case "jump":
                    jump = True
                case _:
                    pass

        self.dx = move * self.speed

        if jump and self.can_jump():
            self.dy = -self.jump_height
            self.grounded = False


class Arena:
    def __init__(self, scale):
        self.configs = config.ARENA_CONFIGS
        self.platforms = []
        self.current = None

        self.spawn_positions = []

        self.scale = scale

        self.blast_zone_margin = 150 * self.scale
        self.blast_zone = (
            -self.blast_zone_margin,
            -self.blast_zone_margin,
            config.arena_size[0] + self.blast_zone_margin,
            config.arena_size[1] + self.blast_zone_margin,
        )

    def load(self, name):
        if name not in self.configs:
            raise ValueError(f"Unknown arena: {name}")
        platforms = self.configs[name]["platforms"]
        self.current = name
        self.platforms = [
            Platform(
                x * self.scale, y * self.scale, width * self.scale, height * self.scale
            )
            for x, y, width, height in platforms
        ]
        spawn_positions = self.configs[name]["spawns"]
        self.spawn_positions = [
            (x * self.scale, y * self.scale) for x, y in spawn_positions
        ]

    def is_out_of_bounds(self, fighter):
        return (
            fighter.right < self.blast_zone[0]
            or fighter.left > self.blast_zone[2]
            or fighter.bottom < self.blast_zone[1]
            or fighter.top > self.blast_zone[3]
        )


class Match:
    def __init__(self, fighters, arena, stocks=2, max_time=240):
        self.stocks = {fighter: stocks for fighter in fighters}
        self.fighters = fighters
        self.arena = arena
        self.eliminated = []  # waiting to respawn
        self.max_time = max_time
        self.elapsed_time = 0.0

    def eliminate(self, fighter):
        if fighter not in self.eliminated:
            self.stocks[fighter] -= 1
            self.eliminated.append(fighter)

    def respawn(self, fighter):
        if fighter in self.eliminated and self.stocks[fighter] > 0:
            self.eliminated.remove(fighter)
            fighter.x, fighter.y = self.get_spawn_position(fighter)
            fighter.grounded = False
            fighter.dx, fighter.dy = (
                0,
                0,
            )

    def check_status(self):
        if self.elapsed_time >= self.max_time:
            return None
        alive = []
        winner = False
        for fighter in self.fighters:
            if self.stocks[fighter] > 0:
                alive.append(fighter)
        if len(alive) == 1:
            winner = alive[0]
        if len(alive) == 0:
            winner = None
        return winner

    def get_spawn_position(self, fighter):
        index = self.fighters.index(fighter)
        return self.arena.spawn_positions[index]

    def tick(self, dt):
        self.elapsed_time += dt
        for fighter in self.fighters:
            if fighter not in self.eliminated:
                fighter.tick(dt, self.arena.platforms)

                if self.arena.is_out_of_bounds(fighter):
                    self.eliminate(fighter)
                    self.respawn(fighter)
        return self.check_status()

import random as r

import config

ATTACKS = {
    "jab": {
        "startup": 2,
        "active": 3,
        "recovery": 7,
        "damage": 3,
        "knockback": 5,
        "lunge": 20,
        "hitbox": (30, 10, 35, 25),
        "hitstun": 10,
        "finisher_multiplier": 1.75,
    },
    "heavy": {
        "startup": 9,
        "active": 5,
        "recovery": 12,
        "damage": 10,
        "knockback": 6,
        "lunge": 60,
        "hitbox": (30, 5, 55, 35),
        "hitstun": 20,
        "finisher_multiplier": 2.0,
    },
}

FPS = 60


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


def rectangles_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


class Fighter:
    def __init__(
        self,
        name="fighter",
        speed=600,
        gravity=2000,
        jump_height=1000,
        x=0,
        y=0,
        width=50,
        height=80,
    ):
        self.name = name

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

        self.facing = 1
        self.attack = None
        self.attack_timer = 0
        self.attack_hit_targets = set()

        self.damage_percent = 0.0
        self.hitstun_timer = 0.0

        self.attack_distance = 0

        self.combo_count = 0
        self.combo_timer = 0.0

        self.unsupported_time = 0.0

        self.DOUBLE_JUMPS = 1
        self.jumps = self.DOUBLE_JUMPS

    def can_jump(self) -> bool:
        return self.grounded or self.jumps > 0

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

    @property
    def hurtbox(self):
        return (self.x, self.y, self.width, self.height)

    def take_hit(self, attacker, is_finisher=False):
        attack = ATTACKS[attacker.attack]

        self.damage_percent += attack["damage"]

        knockback = attack["knockback"] * (1 + self.damage_percent / 300)

        if is_finisher:
            knockback *= attack["finisher_multiplier"]

        self.dx = attacker.facing * knockback * 100
        self.dy = -knockback * 50
        self.grounded = False

        self.hitstun_timer = attack["hitstun"] / FPS

        self.attack = None
        self.attack_timer = 0
        self.attack_distance = 0

        self.combo_count = 0
        self.combo_timer = 0

    def register_combo_hit(self):
        if self.combo_timer <= 0:
            self.combo_count = 0

        self.combo_count += 1
        self.combo_timer = config.COMBO_WINDOW

        if self.combo_count >= config.COMBO_FINISHER_HIT:
            self.combo_count = 0
            self.combo_timer = 0
            return True

        return False

    @property
    def attack_hitbox(self):
        if self.get_attack_phase() != "active":
            return None

        offset_x, offset_y, width, height = ATTACKS[self.attack]["hitbox"]

        if self.facing == 1:
            x = self.right + offset_x
        else:
            x = self.left - offset_x - width

        return (x, self.y + offset_y, width, height)

    def check_hit(self, target):
        hitbox = self.attack_hitbox

        if hitbox is None or target in self.attack_hit_targets:
            return False

        if rectangles_overlap(hitbox, target.hurtbox):
            self.attack_hit_targets.add(target)
            return True

        return False

    def tick(self, dt, platforms):
        if not self.grounded:
            if self.dx > 0:
                self.dx = max(0, self.dx - config.AIR_FRICTION * dt)
            elif self.dx < 0:
                self.dx = min(0, self.dx + config.AIR_FRICTION * dt)

        if self.combo_timer > 0:
            self.combo_timer = max(0, self.combo_timer - dt)

            if self.combo_timer == 0:
                self.combo_count = 0

        if self.hitstun_timer > 0:
            self.hitstun_timer = max(0, self.hitstun_timer - dt)

        elif self.attack is not None:
            self.attack_timer += dt
            phase = self.get_attack_phase()
            attack = ATTACKS[self.attack]

            if phase == "active":
                remaining = attack["lunge"] - self.attack_distance
                distance = min(
                    remaining, attack["lunge"] * dt / (attack["active"] / FPS)
                )

                self.dx = self.facing * distance / dt
                self.attack_distance += distance
            else:
                self.dx = 0

            if phase is None:
                self.attack = None
                self.attack_timer = 0
                self.dx = 0

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
        if self.grounded:
            self.jumps = 1

    def apply_input(self, actions):
        if self.hitstun_timer > 0:
            return
        for action in actions:
            if action in ATTACKS and self.attack is None:
                self.attack = action
                self.attack_timer = 0
                self.attack_hit_targets.clear()
                self.attack_distance = 0
                break
        if self.attack is None:
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

            if move < 0:
                self.facing = -1
            elif move > 0:
                self.facing = 1

            if move != 0 or self.grounded:
                self.dx = move * self.speed

            if jump and self.can_jump():
                if not self.grounded:
                    self.jumps -= 1
                    self.dy = -self.jump_height * 0.5
                else:
                    self.dy = -self.jump_height

                self.grounded = False

    def get_attack_phase(self):
        if self.attack is None:
            return None

        attack = ATTACKS[self.attack]
        timer = self.attack_timer * FPS

        if timer < attack["startup"]:
            return "startup"
        elif timer < attack["startup"] + attack["active"]:
            return "active"
        elif timer < attack["startup"] + attack["active"] + attack["recovery"]:
            return "recovery"

        return None


class Arena:
    def __init__(self, scale):
        self.configs = config.ARENA_CONFIGS
        self.platforms = []
        self.current = None

        self.spawn_positions = []

        self.scale = scale

        self.blast_zone_margin = 300 * self.scale
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

    def has_platform_below(self, fighter):
        for platform in self.platforms:
            horizontally_overlapping = (
                fighter.right > platform.left and fighter.left < platform.right
            )

            if horizontally_overlapping and fighter.bottom <= platform.top:
                return True

        return False

    def is_out_of_bounds(self, fighter):
        return (
            fighter.right < self.blast_zone[0]
            or fighter.left > self.blast_zone[2]
            or fighter.bottom < self.blast_zone[1]
            or fighter.top > self.blast_zone[3]
        )


class Match:
    def __init__(self, fighters, arena, stocks=1, max_time=240):
        self.stocks = {fighter: stocks for fighter in fighters}
        self.fighters = fighters
        self.arena = arena
        self.eliminated = []  # waiting to respawn
        self.max_time = max_time
        self.elapsed_time = 0.0

        self.events = []

    def start(self):
        for fighter in self.fighters:
            fighter.x, fighter.y = self.get_spawn_position(fighter)
            fighter.dx = 0
            fighter.dy = 0
            fighter.grounded = False

    def eliminate(self, fighter):
        if fighter not in self.eliminated:
            self.stocks[fighter] -= 1
            self.emit_event(
                "fighter_eliminated",
                fighter=fighter,
                stocks_remaining=self.stocks[fighter],
            )
            self.eliminated.append(fighter)

    def emit_event(self, event_type, **data):
        self.events.append(
            {
                "type": event_type,
                **data,
            }
        )

    def respawn(self, fighter):
        if fighter in self.eliminated and self.stocks[fighter] > 0:
            self.eliminated.remove(fighter)
            fighter.x, fighter.y = self.get_spawn_position(fighter)
            fighter.grounded = False
            fighter.dx, fighter.dy = (
                0,
                0,
            )

            fighter.damage_percent = 0.0
            fighter.hitstun_timer = 0.0

            fighter.attack = None
            fighter.attack_timer = 0
            fighter.attack_distance = 0
            fighter.attack_hit_targets.clear()

    def check_collisions(self):
        for fighter in self.fighters:
            if fighter in self.eliminated:
                continue

            for target in self.fighters:
                if target == fighter or target in self.eliminated:
                    continue

                if fighter.check_hit(target):
                    attack = fighter.attack
                    is_finisher = fighter.register_combo_hit()
                    target.take_hit(fighter, is_finisher)

                    self.emit_event(
                        "attack_hit",
                        attacker=fighter,
                        target=target,
                        attack=attack,
                        combo_count=(
                            config.COMBO_FINISHER_HIT
                            if is_finisher
                            else fighter.combo_count
                        ),
                        is_finisher=is_finisher,
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
        self.events.clear()
        self.elapsed_time += dt
        for fighter in self.fighters:
            if fighter not in self.eliminated:
                fighter.tick(dt, self.arena.platforms)

                if self.arena.is_out_of_bounds(fighter):
                    self.eliminate(fighter)
                    self.respawn(fighter)
            if self.arena.has_platform_below(fighter):
                fighter.unsupported_time = 0.0
            else:
                fighter.unsupported_time += dt

        self.check_collisions()
        return self.check_status()

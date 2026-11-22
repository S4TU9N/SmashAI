ARENA_SCALE = 1.5
ARENA_WIDTH = 1280
ARENA_HEIGHT = 780
RENDER = True

MATCH_MAX_TIME = 240  # seconds

arena_size = (ARENA_WIDTH * ARENA_SCALE, ARENA_HEIGHT * ARENA_SCALE)

ARENA_CONFIGS = {
    "battlefield": {
        "platforms": [
            (200, 600, 880, 30),
            (350, 450, 200, 20),
            (730, 450, 200, 20),
        ],
        "spawns": [(400, 400), (800, 400)],
    },
    "final_destination": {
        "platforms": [
            (200, 600, 880, 30),
        ],
        "spawns": [(400, 400), (800, 400)],
    },
}


COMBO_WINDOW = 1.0
COMBO_FINISHER_HIT = 3

AIR_FRICTION = 300


# rewards
JAB_REWARD = 1.0
HEAVY_REWARD_MULTIPLIER = 1.25
COMBO_REWARD_INCREMENT = 0.5
COMBO_FINISHER_REWARD = 3.0

ELIMINATION_REWARD = 10.0
ELIMINATION_PENALTY = -10.0

UNSUPPORTED_GRACE_PERIOD = 0.25
UNSUPPORTED_PENALTY_RATE = 1.0
UNSUPPORTED_PENALTY_CAP = 1.0

WIN_REWARD = 10
LOSS_REWARD = -10
DRAW_REWARD = -2

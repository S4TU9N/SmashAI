import numpy as np

import config

ATTACK_PHASES = {
    None: 0,
    "startup": 1,
    "active": 2,
    "recovery": 3,
}


def get_observation(fighter, opponent, arena):
    width, height = config.arena_size

    relative_x = (opponent.x - fighter.x) / width
    relative_y = (opponent.y - fighter.y) / height

    relative_dx = (opponent.dx - fighter.dx) / (fighter.speed * 2)
    relative_dy = (opponent.dy - fighter.dy) / (fighter.speed * 2)

    own_edge_distance = (
        min(
            fighter.x - arena.blast_zone[0],
            arena.blast_zone[2] - fighter.right,
        )
        / width
    )

    opponent_edge_distance = (
        min(
            opponent.x - arena.blast_zone[0],
            arena.blast_zone[2] - opponent.right,
        )
        / width
    )

    has_platform = float(arena.has_platform_below(fighter))
    platform_distance = get_nearest_platform_distance(fighter, arena)

    remaining_jumps = fighter.jumps / fighter.DOUBLE_JUMPS

    return np.array(
        [
            relative_x,
            relative_y,
            relative_dx,
            relative_dy,
            float(fighter.grounded),
            float(opponent.grounded),
            fighter.damage_percent / 100,
            opponent.damage_percent / 100,
            fighter.facing,
            min(fighter.hitstun_timer, 1.0),
            min(opponent.hitstun_timer, 1.0),
            ATTACK_PHASES[fighter.get_attack_phase()],
            ATTACK_PHASES[opponent.get_attack_phase()],
            own_edge_distance,
            opponent_edge_distance,
            has_platform,
            platform_distance,
            remaining_jumps,
        ],
        dtype=np.float32,
    )


def get_nearest_platform_distance(fighter, arena):
    nearest_distance = float("inf")

    for platform in arena.platforms:
        dx = max(
            platform.left - fighter.right,
            fighter.left - platform.right,
            0,
        )

        dy = abs(platform.top - fighter.bottom)

        distance = (dx**2 + dy**2) ** 0.5
        nearest_distance = min(nearest_distance, distance)

    if nearest_distance == float("inf"):
        return 1.0

    return min(nearest_distance / config.arena_size[0], 1.0)

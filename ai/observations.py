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
        ],
        dtype=np.float32,
    )

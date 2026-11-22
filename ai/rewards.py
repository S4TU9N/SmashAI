import config


def calculate_reward(fighter, events, dt):
    reward = 0.0

    unsupported_time = fighter.unsupported_time - config.UNSUPPORTED_GRACE_PERIOD

    if unsupported_time > 0:
        penalty = (
            min(
                unsupported_time,
                config.UNSUPPORTED_PENALTY_CAP,
            )
            * config.UNSUPPORTED_PENALTY_RATE
            * dt
        )

        reward -= penalty

    attack_multiplier = {
        "jab": 1.0,
        "heavy": 1.25,
    }

    for event in events:
        event_type = event["type"]

        if event_type == "attack_hit":
            if event["attacker"] is not fighter:
                continue

            combo_count = event["combo_count"]
            is_finisher = event["is_finisher"]

            hit_reward = (
                config.JAB_REWARD + (combo_count - 1) * config.COMBO_REWARD_INCREMENT
            )

            if is_finisher:
                hit_reward = config.COMBO_FINISHER_REWARD

            reward += hit_reward * attack_multiplier[event["attack"]]

        elif event_type == "fighter_eliminated":
            if event["fighter"] is fighter:
                reward -= 10.0
            else:
                reward += 10.0

    return reward

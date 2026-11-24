def calculate_reward(fighter, events, dt, agent_config):
    reward = 0.0

    unsupported_time = fighter.unsupported_time - agent_config.UNSUPPORTED_GRACE_PERIOD

    if unsupported_time > 0:
        penalty = (
            min(
                unsupported_time,
                agent_config.UNSUPPORTED_PENALTY_CAP,
            )
            * agent_config.UNSUPPORTED_PENALTY_RATE
            * dt
        )

        reward -= penalty

    attack_multiplier = {
        "jab": 1.0,
        "heavy": agent_config.HEAVY_REWARD_MULTIPLIER,
    }

    for event in events:
        event_type = event["type"]

        if event_type == "attack_hit":
            if event["attacker"] is not fighter:
                continue

            combo_count = event["combo_count"]
            is_finisher = event["is_finisher"]

            hit_reward = (
                agent_config.JAB_REWARD
                + (combo_count - 1) * agent_config.COMBO_REWARD_INCREMENT
            )

            if is_finisher:
                hit_reward = agent_config.COMBO_FINISHER_REWARD

            reward += hit_reward * attack_multiplier[event["attack"]]

        elif event_type == "fighter_eliminated":
            if event["fighter"] is fighter:
                reward += agent_config.ELIMINATION_PENALTY
            else:
                reward += agent_config.ELIMINATION_REWARD

    return reward

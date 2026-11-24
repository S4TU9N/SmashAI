from pathlib import Path

import torch

from .checkpoint import read_checkpoint
from .environment import SmashEnvironment
from .policy import PolicyNetwork


def evaluate(checkpoint_path, episodes, agent_config):
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    checkpoint = read_checkpoint(checkpoint_path)

    policy = PolicyNetwork()
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()

    env = SmashEnvironment(agent_config, render_mode=None)

    results = {
        "win": 0,
        "loss": 0,
        "draw": 0,
    }

    total_agent_damage = 0.0
    total_opponent_damage = 0.0
    total_duration = 0.0

    try:
        for _ in range(episodes):
            observation, _ = env.reset()

            terminated = False
            truncated = False

            while not terminated and not truncated:
                observation_tensor = torch.tensor(
                    observation,
                    dtype=torch.float32,
                )

                with torch.no_grad():
                    logits = policy(observation_tensor)
                    action = (logits >= 0).float().numpy()

                observation, _, terminated, truncated, info = env.step(action)

            results[info["result"]] += 1
            total_agent_damage += info["agent_damage"]
            total_opponent_damage += info["opponent_damage"]
            total_duration += info["match_duration"]

    finally:
        env.close()

    print(f"\nevaluation: {episodes} episodes")
    print("-" * 30)

    for result, count in results.items():
        percentage = count / episodes * 100
        print(f"{result}: {count} ({percentage:.1f}%)")

    print("-" * 30)
    print(f"average agent damage: {total_agent_damage / episodes:.1f}%")
    print(f"average opponent damage: {total_opponent_damage / episodes:.1f}%")
    print(f"average duration: {total_duration / episodes:.1f}s")

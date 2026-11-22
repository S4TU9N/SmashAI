import numpy as np
from gymnasium.utils.env_checker import check_env

from ai.environment import SmashEnvironment

env = SmashEnvironment()
check_env(env)
print("Environment validation passed")

for episode in range(3):
    observation, info = env.reset(seed=episode)

    total_reward = 0
    steps = 0

    while True:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        assert observation.shape == (18,)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)

        if terminated or truncated:
            break

    print(f"Episode: {episode}")
    print(f"Steps: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Winner: {env.match.check_status()}")
    print()

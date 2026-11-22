import gymnasium as gym

from ai.environment import SmashEnvironment

env = SmashEnvironment(render_mode="realtime")

observation, info = env.reset()

try:
    while True:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode finished. Reward:", reward)
            break
finally:
    env.close()

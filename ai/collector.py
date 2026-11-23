import torch

from ai.policy import sample_action


def collect_rollout(env, policy, value, buffer, observation, steps=2048):
    completed_episodes = []
    for _ in range(steps):
        observation_tensor = torch.tensor(
            observation,
            dtype=torch.float32,
        )

        with torch.no_grad():
            logits = policy(observation_tensor)
            value_estimate = value(observation_tensor)
            action, log_probability = sample_action(logits)

        next_observation, reward, terminated, truncated, info = env.step(action.numpy())

        next_observation_tensor = torch.tensor(
            next_observation,
            dtype=torch.float32,
        )

        with torch.no_grad():
            next_value = value(next_observation_tensor)

        buffer.add(
            observation,
            action,
            log_probability,
            reward,
            value_estimate,
            next_value,
            terminated,
            truncated,
        )

        if terminated or truncated:
            observation, _ = env.reset()
            completed_episodes.append(info)
        else:
            observation = next_observation

    return buffer, observation, completed_episodes

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probabilities = []
        self.rewards = []
        self.values = []
        self.next_values = []
        self.terminated = []
        self.truncated = []

    def add(
        self,
        observation,
        action,
        log_probability,
        reward,
        value,
        next_value,
        terminated,
        truncated,
    ):
        self.observations.append(observation.copy())
        self.actions.append(action.detach().cpu().numpy())
        self.log_probabilities.append(log_probability.item())
        self.rewards.append(reward)
        self.values.append(value.item())
        self.next_values.append(next_value.item())
        self.terminated.append(terminated)
        self.truncated.append(truncated)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probabilities.clear()
        self.rewards.clear()
        self.values.clear()
        self.next_values.clear()
        self.terminated.clear()
        self.truncated.clear()

    def get_tensors(self):
        if not self.observations:
            raise ValueError("cannot get tensors from an empty buffer")

        observations = torch.tensor(
            np.array(self.observations),
            dtype=torch.float32,
        )  # convert observations and actions to np.array following pytorchs recommendation
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32)
        log_probabilities = torch.tensor(self.log_probabilities, dtype=torch.float32)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        next_values = torch.tensor(self.next_values, dtype=torch.float32)
        terminated = torch.tensor(self.terminated, dtype=torch.bool)
        truncated = torch.tensor(self.truncated, dtype=torch.bool)

        return (
            observations,
            actions,
            log_probabilities,
            rewards,
            values,
            next_values,
            terminated,
            truncated,
        )

    def get_batches(self, batch_size, advantages, returns):
        tensors = self.get_tensors()

        indices = torch.randperm(len(self.observations))

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]

            yield tuple(tensor[batch_indices] for tensor in tensors) + (
                advantages[batch_indices],
                returns[batch_indices],
            )


def calculate_gae(
    rewards,
    values,
    next_values,
    terminated,
    truncated,
    gamma=0.99,
    gae_lambda=0.95,
):
    advantages = [0.0] * len(rewards)
    next_advantage = 0.0

    for t in reversed(range(len(rewards))):
        next_value = next_values[t]

        if terminated[t]:
            next_value = 0.0

        delta = rewards[t] + gamma * next_value - values[t]

        if terminated[t] or truncated[t]:
            next_advantage = 0.0

        advantages[t] = delta + gamma * gae_lambda * next_advantage
        next_advantage = advantages[t]

    return advantages

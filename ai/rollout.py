import numpy as np
import torch


class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probabilities = []
        self.rewards = []
        self.values = []
        self.terminated = []
        self.truncated = []

    def add(
        self,
        observation,
        action,
        log_probability,
        reward,
        value,
        terminated,
        truncated,
    ):
        self.observations.append(observation.copy())
        self.actions.append(action.detach().cpu().numpy())
        self.log_probabilities.append(log_probability.item())
        self.rewards.append(reward)
        self.values.append(value.item())
        self.terminated.append(terminated)
        self.truncated.append(truncated)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probabilities.clear()
        self.rewards.clear()
        self.values.clear()
        self.terminated.clear()
        self.truncated.clear()

    def get_tensors(self):
        if not self.observations:
            raise ValueError("cannot get tensors from an empty buffer")

        observations = torch.tensor(
            np.array(self.observations),
            dtype=torch.float32,
        )
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32)
        log_probabilities = torch.tensor(self.log_probabilities, dtype=torch.float32)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        terminated = torch.tensor(self.terminated, dtype=torch.bool)
        truncated = torch.tensor(self.truncated, dtype=torch.bool)

        return (
            observations,
            actions,
            log_probabilities,
            rewards,
            values,
            terminated,
            truncated,
        )

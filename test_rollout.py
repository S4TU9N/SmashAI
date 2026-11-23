import numpy as np
import torch

from ai.policy import PolicyNetwork
from ai.ppo import create_optimizers, train
from ai.rollout import RolloutBuffer
from ai.value import ValueNetwork

buffer = RolloutBuffer()

for _ in range(128):
    observation = np.random.randn(18).astype(np.float32)

    with torch.no_grad():
        observation_tensor = torch.tensor(observation)
        logits = policy(observation_tensor)
        action, log_probability = sample_action(logits)
        value_estimate = value(observation_tensor)

    buffer.add(
        observation,
        action,
        log_probability,
        np.random.randn(),
        value_estimate,
        torch.tensor(0.0),
        False,
        False,
    )

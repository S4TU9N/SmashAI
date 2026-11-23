import numpy as np
import torch

from ai.policy import PolicyNetwork, sample_action
from ai.ppo import create_optimizers, train
from ai.rollout import RolloutBuffer
from ai.value import ValueNetwork

buffer = RolloutBuffer()
policy = PolicyNetwork()
value = ValueNetwork()

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

policy_optimizer, value_optimizer = create_optimizers(policy, value)

initial_policy = [p.clone() for p in policy.parameters()]
initial_value = [p.clone() for p in value.parameters()]

policy_loss, value_loss = train(
    buffer,
    policy,
    value,
    policy_optimizer,
    value_optimizer,
)

print("policy loss:", policy_loss)
print("value loss:", value_loss)

print(
    "policy changed:",
    any(
        not torch.equal(old, new)
        for old, new in zip(initial_policy, policy.parameters())
    ),
)

print(
    "value changed:",
    any(
        not torch.equal(old, new) for old, new in zip(initial_value, value.parameters())
    ),
)

print("buffer empty:", len(buffer.observations) == 0)

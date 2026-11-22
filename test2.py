import torch

from ai.policy import PolicyNetwork, sample_action
from ai.value import ValueNetwork

policy = PolicyNetwork()
observation = torch.randn(18)

critic = ValueNetwork()
value = critic(observation)

print("value:", value)
print("shape:", value.shape)

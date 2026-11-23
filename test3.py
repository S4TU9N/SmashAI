import torch

from ai.policy import PolicyNetwork, evaluate_actions

policy = PolicyNetwork()

observations = torch.rand(4, 18)
actions = torch.randint(0, 2, (4, 5)).float()

logits = policy(observations)

log_probabilities, entropy = evaluate_actions(logits, actions)

print("logits:", logits.shape)
print("log probabilities:", log_probabilities.shape)
print("entropy:", entropy.shape)

print("log probabilities:", log_probabilities)
print("entropy:", entropy)

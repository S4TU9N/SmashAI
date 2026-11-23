import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(self, observation):
        return self.network(observation)


def sample_action(logits):
    distribution = torch.distributions.Bernoulli(logits=logits)

    action = distribution.sample()
    log_probability = distribution.log_prob(action).sum()

    return action, log_probability


def evaluate_actions(logits, actions):
    distribution = torch.distributions.Bernoulli(logits=logits)

    log_probabilities = distribution.log_prob(actions).sum(dim=-1)
    entropy = distribution.entropy().sum(dim=-1)

    return log_probabilities, entropy

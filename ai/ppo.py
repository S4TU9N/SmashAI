import torch
from torch import optim

from ai.policy import evaluate_actions
from ai.rollout import calculate_gae


def calculate_policy_loss(
    new_log_probabilities,
    old_log_probabilities,
    advantages,
    entropy,
    clip_epsilon=0.2,
    entropy_coefficient=0.01,
):
    ratio = torch.exp(new_log_probabilities - old_log_probabilities)

    unclipped = ratio * advantages

    clipped = (
        torch.clamp(
            ratio,
            1 - clip_epsilon,
            1 + clip_epsilon,
        )
        * advantages
    )

    loss = -torch.minimum(unclipped, clipped).mean()

    total_loss = loss - (entropy_coefficient * entropy.mean())

    return total_loss


def calculate_value_loss(values, returns):
    return torch.nn.functional.mse_loss(values, returns)


def create_optimizers(policy, value, learning_rate=3e-4):
    policy_optimizer = optim.Adam(
        policy.parameters(),
        lr=learning_rate,
    )

    value_optimizer = optim.Adam(
        value.parameters(),
        lr=learning_rate,
    )

    return policy_optimizer, value_optimizer


def calculate_total_loss(policy_loss, value_loss, value_coefficient=0.5):
    return policy_loss + value_coefficient * value_loss


def update_minibatch(
    policy,
    value,
    policy_optimizer,
    value_optimizer,
    observations,
    actions,
    old_log_probabilities,
    advantages,
    returns,
):
    # policy forward pass
    logits = policy(observations)

    new_log_probabilities, entropy = evaluate_actions(
        logits,
        actions,
    )

    policy_loss = calculate_policy_loss(
        new_log_probabilities,
        old_log_probabilities,
        advantages,
        entropy,
    )

    # value forward pass
    values = value(observations).squeeze(-1)

    value_loss = calculate_value_loss(values, returns)

    # policy update
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # value update
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    return policy_loss.item(), value_loss.item()


def normalize_advantages(advantages):
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def train(
    buffer,
    policy,
    value,
    policy_optimizer,
    value_optimizer,
    epochs=4,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
):
    (
        observations,
        actions,
        old_log_probabilities,
        rewards,
        values,
        next_values,
        terminated,
        truncated,
    ) = buffer.get_tensors()

    advantages = calculate_gae(
        rewards.tolist(),
        values.tolist(),
        next_values.tolist(),
        terminated.tolist(),
        truncated.tolist(),
        gamma,
        gae_lambda,
    )

    returns = (
        torch.tensor(
            advantages,
            dtype=torch.float32,
        )
        + values
    )

    advantages = torch.tensor(
        advantages,
        dtype=torch.float32,
    )

    advantages = normalize_advantages(advantages)

    policy_losses = []
    value_losses = []

    for epoch in range(epochs):
        for batch in buffer.get_batches(batch_size, advantages, returns):
            (
                batch_observations,
                batch_actions,
                batch_old_log_probabilities,
                batch_rewards,
                batch_values,
                batch_next_values,
                batch_terminated,
                batch_truncated,
                batch_advantages,
                batch_returns,
            ) = batch

            policy_loss, value_loss = update_minibatch(
                policy,
                value,
                policy_optimizer,
                value_optimizer,
                batch_observations,
                batch_actions,
                batch_old_log_probabilities,
                batch_advantages,
                batch_returns,
            )

            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

    average_policy_loss = sum(policy_losses) / len(policy_losses)
    average_value_loss = sum(value_losses) / len(value_losses)

    buffer.clear()

    return average_policy_loss, average_value_loss

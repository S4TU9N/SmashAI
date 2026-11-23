from pathlib import Path

import torch


def save_checkpoint(
    path, policy, value, policy_optimizer, value_optimizer, metadata=None
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "policy_optimizer": policy_optimizer.state_dict(),
            "value_optimizer": value_optimizer.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(path, policy, value, policy_optimizer, value_optimizer):
    checkpoint = torch.load(path, weights_only=True)

    policy.load_state_dict(checkpoint["policy"])
    value.load_state_dict(checkpoint["value"])

    policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
    value_optimizer.load_state_dict(checkpoint["value_optimizer"])

    return checkpoint

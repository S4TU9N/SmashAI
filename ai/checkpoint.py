import os
from pathlib import Path

import torch


def save_checkpoint(
    path,
    policy,
    value,
    policy_optimizer,
    value_optimizer,
    metadata=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(path.name + ".tmp")

    torch.save(
        {
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "policy_optimizer": policy_optimizer.state_dict(),
            "value_optimizer": value_optimizer.state_dict(),
            "metadata": metadata or {},
        },
        temp_path,
    )

    os.replace(temp_path, path)


def load_checkpoint(path, policy, value, policy_optimizer, value_optimizer):
    checkpoint = read_checkpoint(path)

    policy.load_state_dict(checkpoint["policy"])
    value.load_state_dict(checkpoint["value"])

    policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
    value_optimizer.load_state_dict(checkpoint["value_optimizer"])

    return checkpoint


def read_checkpoint(path):
    return torch.load(path, weights_only=True)

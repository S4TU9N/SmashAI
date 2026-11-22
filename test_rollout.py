import numpy as np
import torch

from ai.rollout import RolloutBuffer

buffer = RolloutBuffer()

observation = np.zeros(18, dtype=np.float32)
action = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
log_probability = torch.tensor(-2.5)
value = torch.tensor(0.75)

buffer.add(
    observation,
    action,
    log_probability,
    reward=1.5,
    value=value,
    terminated=False,
    truncated=False,
)

tensors = buffer.get_tensors()

for tensor in tensors:
    print(tensor.shape, tensor.dtype, tensor)

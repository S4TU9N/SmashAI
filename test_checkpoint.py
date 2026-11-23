import torch

from ai.checkpoint import load_checkpoint, save_checkpoint
from ai.policy import PolicyNetwork
from ai.ppo import create_optimizers
from ai.value import ValueNetwork

policy = PolicyNetwork()
value = ValueNetwork()

policy_optimizer, value_optimizer = create_optimizers(
    policy,
    value,
)

for parameter in policy.parameters():
    parameter.data.add_(1.0)

for parameter in value.parameters():
    parameter.data.add_(1.0)

dummy_input = torch.randn(18)

policy_loss = policy(dummy_input).sum()
value_loss = value(dummy_input).sum()

policy_optimizer.zero_grad()
policy_loss.backward()
policy_optimizer.step()

value_optimizer.zero_grad()
value_loss.backward()
value_optimizer.step()

save_checkpoint(
    "checkpoints/test.pt",
    policy,
    value,
    policy_optimizer,
    value_optimizer,
)

print("checkpoint saved")


new_policy = PolicyNetwork()
new_value = ValueNetwork()

new_policy_optimizer, new_value_optimizer = create_optimizers(
    new_policy,
    new_value,
)

load_checkpoint(
    "checkpoints/test.pt",
    new_policy,
    new_value,
    new_policy_optimizer,
    new_value_optimizer,
)

for original, restored in zip(policy.parameters(), new_policy.parameters()):
    print(torch.equal(original, restored))


for original, restored in zip(value.parameters(), new_value.parameters()):
    print(torch.equal(original, restored))

original_state = policy_optimizer.state_dict()
restored_state = new_policy_optimizer.state_dict()

for key in original_state["state"]:
    for state_name in original_state["state"][key]:
        original = original_state["state"][key][state_name]
        restored = restored_state["state"][key][state_name]

        if isinstance(original, torch.Tensor):
            print(state_name, torch.equal(original, restored))
        else:
            print(state_name, original == restored)

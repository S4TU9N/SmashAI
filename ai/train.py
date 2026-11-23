from pathlib import Path

from ai.checkpoint import load_checkpoint, save_checkpoint
from ai.collector import collect_rollout
from ai.environment import SmashEnvironment
from ai.policy import PolicyNetwork
from ai.ppo import create_optimizers, train
from ai.rollout import RolloutBuffer
from ai.value import ValueNetwork


def main():
    env = None

    policy = PolicyNetwork()
    value = ValueNetwork()

    policy_optimizer, value_optimizer = create_optimizers(
        policy,
        value,
    )

    checkpoint_path = Path("checkpoints/main/latest.pt")
    history_path = Path("checkpoints/main/history")

    iteration = 0
    total_timesteps = 0

    if checkpoint_path.exists():
        print("loaded existing checkpoint")
        checkpoint = load_checkpoint(
            checkpoint_path,
            policy,
            value,
            policy_optimizer,
            value_optimizer,
        )

        metadata = checkpoint.get("metadata", {})
        iteration = metadata.get("iteration", 0)
        total_timesteps = metadata.get("total_timesteps", 0)
    else:
        print("starting new training run")

    buffer = RolloutBuffer()

    try:
        env = SmashEnvironment(render_mode=None)
        observation, _ = env.reset()

        while True:
            buffer, observation, completed_episodes = collect_rollout(
                env,
                policy,
                value,
                buffer,
                observation,
            )

            for episode in completed_episodes:
                print(
                    f"match: {episode['result']} | "
                    f"duration: {episode['match_duration']:.1f}s"
                )

            iteration += 1
            total_timesteps += 2048

            policy_loss, value_loss = train(
                buffer,
                policy,
                value,
                policy_optimizer,
                value_optimizer,
            )

            print(
                f"iteration: {iteration} | "
                f"policy loss: {policy_loss:.4f} | "
                f"value loss: {value_loss:.4f} | "
                f"timesteps: {total_timesteps}"
            )

            save_checkpoint(
                checkpoint_path,
                policy,
                value,
                policy_optimizer,
                value_optimizer,
                metadata={
                    "agent": "main",
                    "iteration": iteration,
                    "total_timesteps": total_timesteps,
                },
            )

            if iteration % 50 == 0:
                save_checkpoint(
                    history_path / f"iteration-{iteration:06d}.pt",
                    policy,
                    value,
                    policy_optimizer,
                    value_optimizer,
                    metadata={
                        "agent": "main",
                        "iteration": iteration,
                        "total_timesteps": total_timesteps,
                    },
                )
                print(f"saved historical checkpoint: iteration {iteration}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

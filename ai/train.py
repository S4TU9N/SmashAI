import json
import signal
from collections import deque
from pathlib import Path

from ai.checkpoint import load_checkpoint, save_checkpoint
from ai.collector import collect_rollout
from ai.environment import SmashEnvironment
from ai.policy import PolicyNetwork
from ai.ppo import create_optimizers, train
from ai.rollout import RolloutBuffer
from ai.value import ValueNetwork


def main(agent_name, agent_config, iterations=None, quiet=False, silent=False):
    env = None

    if iterations is not None and iterations < 1:
        raise ValueError("iterations must be at least 1")

    policy = PolicyNetwork()
    value = ValueNetwork()

    policy_optimizer, value_optimizer = create_optimizers(
        policy,
        value,
    )

    agent_path = Path("checkpoints") / agent_name

    checkpoint_path = agent_path / "latest.pt"
    history_path = agent_path / "history"
    episode_log_path = agent_path / "episodes.jsonl"

    iteration = 0
    total_timesteps = 0

    recent_episodes = deque(maxlen=100)

    if episode_log_path.exists():
        with episode_log_path.open() as file:
            for line in file:
                recent_episodes.append(json.loads(line))

    if checkpoint_path.exists():
        if not silent:
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
        if not silent:
            print("starting new training run")

    target_iteration = iteration + iterations if iterations is not None else None

    buffer = RolloutBuffer()

    stop_requested = False

    def handle_sigint(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        env = SmashEnvironment(agent_config, render_mode=None)
        observation, _ = env.reset()

        win_rate = 0.0

        while not stop_requested and (
            target_iteration is None or iteration < target_iteration
        ):
            if stop_requested and not silent:
                print("training interrupted; checkpoint saved")

            buffer, observation, completed_episodes = collect_rollout(
                env,
                policy,
                value,
                buffer,
                observation,
            )

            for episode in completed_episodes:
                recent_episodes.append(episode)

                with episode_log_path.open("a") as file:
                    file.write(json.dumps(episode) + "\n")

                if not quiet and not silent:
                    print(
                        f"match: {episode['result']} | "
                        f"duration: {episode['match_duration']:.1f}s"
                    )

            if recent_episodes:
                win_rate = sum(
                    episode["result"] == "win" for episode in recent_episodes
                ) / len(recent_episodes)

                average_duration = sum(
                    episode["match_duration"] for episode in recent_episodes
                ) / len(recent_episodes)

                average_damage_dealt = sum(
                    episode["opponent_damage"] for episode in recent_episodes
                ) / len(recent_episodes)

                average_damage_received = sum(
                    episode["agent_damage"] for episode in recent_episodes
                ) / len(recent_episodes)

            iteration += 1
            total_timesteps += 2048

            policy_loss, value_loss = train(
                buffer,
                policy,
                value,
                policy_optimizer,
                value_optimizer,
            )

            if stop_requested and not silent:
                print("training interrupted; checkpoint saved")

            if not silent:
                if quiet:
                    print(
                        f"{iteration} | "
                        f"win: {win_rate:.0%} | "
                        f"loss: {policy_loss:.3f}/{value_loss:.3f}"
                    )
                else:
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
                    "agent": agent_name,
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
                        "agent": agent_name,
                        "iteration": iteration,
                        "total_timesteps": total_timesteps,
                    },
                )
                if not silent:
                    print(f"saved historical checkpoint: iteration {iteration}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    from ai.agents.main import config as agent_config

    main("main", agent_config)

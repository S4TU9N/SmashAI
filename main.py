import argparse
from importlib import import_module
from pathlib import Path

from ai.checkpoint import read_checkpoint
from ai.evaluate import evaluate
from ai.train import main as train


def discover_agents():
    agents_path = Path(__file__).parent / "ai" / "agents"

    return sorted(
        directory.name
        for directory in agents_path.iterdir()
        if directory.is_dir() and (directory / "config.py").exists()
    )


def load_agent_config(agent_name):
    if agent_name not in discover_agents():
        raise ValueError(f"unknown agent: {agent_name}")

    return import_module(f"ai.agents.{agent_name}.config")


def list_agents():
    agents = discover_agents()

    if not agents:
        print("no agents found")
        return

    print("available agents:")

    for agent in agents:
        print(f"- {agent}")


def main():
    parser = argparse.ArgumentParser(prog="main")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--agent", default="main")

    subparsers.add_parser("list")

    evaluate_parser = subparsers.add_parser("evaluate")

    checkpoint_group = evaluate_parser.add_mutually_exclusive_group(required=True)

    checkpoint_group.add_argument("--agent")
    checkpoint_group.add_argument("--checkpoint", type=Path)

    evaluate_parser.add_argument("--episodes", type=int, default=100)

    args = parser.parse_args()

    if args.command == "list":
        list_agents()

    elif args.command == "train":
        agent_config = load_agent_config(args.agent)
        train(args.agent, agent_config)

    elif args.command == "evaluate":
        if args.agent:
            agent_name = args.agent
            checkpoint_path = Path("checkpoints") / agent_name / "latest.pt"
        else:
            checkpoint_path = args.checkpoint
            checkpoint = read_checkpoint(checkpoint_path)
            agent_name = checkpoint["metadata"]["agent"]

        agent_config = load_agent_config(agent_name)

        evaluate(
            checkpoint_path,
            args.episodes,
            agent_config,
        )


if __name__ == "__main__":
    main()

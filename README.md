# SmashAI

A simple 2D platform-fighting game inspired by Super Smash Bros., with a custom simulation and reinforcement-learning framework. It uses Proximal Policy Optimization (PPO) to train AI fighters, with configurable rewards, opponents, and arenas.

## Features

- 2D platform fighting built with pygame-ce
- Reinforcement learning using PPO and PyTorch
- Configurable AI reward systems
- Multiple arenas and opponent types
- CLI for training, evaluating, and managing agents

## Installation

### Requirements

- Python
- pygame-ce
- numpy
- gymnasium
- torch

### Setup

Clone the repository:

```bash
git clone <repository-url>
cd SmashAI
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

**Fish:**

```fish
source .venv/bin/activate.fish
```

**Bash:**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The project provides a command-line interface through `main.py`. It is currently intended to be run directly from its Python source rather than installed as a standalone application.

Commands follow this format:

```bash
python -m main <command>
```

### Training

Train an agent:

```bash
python -m main train --agent main --iterations 100
```

- `--agent`: Agent to train (defaults to `main`).
- `--iterations`: Number of additional training iterations. Each iteration collects 2,048 timesteps.

### Evaluation

Evaluate an existing agent:

```bash
python -m main evaluate --agent main --episodes 100
```

- `--agent`: Agent whose latest checkpoint should be evaluated.
- `--checkpoint`: Alternatively, provide a specific checkpoint.
- `--episodes`: Number of matches to evaluate (defaults to 100).
- `--opponent`: Opponent type: `basic`, `aggressive`, or `defensive`.
- `--render-mode`: Rendering mode:
    - Empty: No rendering.
    - `render`: Render without a frame-rate limit.
    - `realtime`: Render at 60 FPS.

### Listing agents

List available agents and their checkpoint status:

```bash
python -m main list
```

### Output options

Training and evaluation support two additional output flags:

- `-q`, `--quiet`: Reduce output.
- `-Q`, `--silent`: Suppress normal output.

All commands also support `-h` or `--help` for additional options.

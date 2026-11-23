import gymnasium as gym
import numpy as np

import config
from simulation import Arena, Fighter, Match

from .observations import get_observation
from .opponents import BasicOpponent
from .rewards import calculate_reward


class SmashEnvironment(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        self.renderer = None

        if render_mode is not None:
            from renderer import Renderer

            self.renderer = Renderer(None)

        self.arena = None
        self.fighter = None
        self.opponent = None
        self.match = None

        self.action_space = gym.spaces.MultiBinary(5)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32,
        )

    def decode_action(self, action):
        names = [
            "move_left",
            "move_right",
            "jump",
            "jab",
            "heavy",
        ]

        return {name for name, pressed in zip(names, action) if pressed}

    def step(self, action):
        dt = 1 / 60

        agent_actions = self.decode_action(action)
        opponent_actions = self.opponent_controller.get_actions(
            self.opponent,
            self.fighter,
        )

        self.fighter.apply_input(agent_actions)
        self.opponent.apply_input(opponent_actions)

        result = self.match.tick(dt)

        reward = calculate_reward(
            self.fighter,
            self.match.events,
            dt,
        )

        if self.renderer:
            self.renderer.poll_events()

            if self.renderer.running:
                self.renderer.draw(self.match.fighters, self.match)

                fps = 0 if self.render_mode == "rendered" else 60
                self.renderer.tick(fps)

        terminated = (
            result is not False and self.match.elapsed_time < self.match.max_time
        )
        truncated = self.match.elapsed_time >= self.match.max_time

        if terminated or truncated:
            if result is self.fighter:
                reward += config.WIN_REWARD
            elif result is self.opponent:
                reward += config.LOSS_REWARD
            else:
                reward += config.DRAW_REWARD

        observation = get_observation(
            self.fighter,
            self.opponent,
            self.arena,
        )

        info = {}

        if terminated or truncated:
            if result is self.fighter:
                info["result"] = "win"
            elif result is self.opponent:
                info["result"] = "loss"
            else:
                info["result"] = "draw"

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.arena = Arena(scale=config.ARENA_SCALE)
        self.fighter = Fighter(name="agent")
        self.opponent = Fighter(name="opponent")
        self.opponent_controller = BasicOpponent()

        self.match = Match(
            fighters=[self.fighter, self.opponent],
            arena=self.arena,
            stocks=1,
            max_time=config.MATCH_MAX_TIME,
        )

        arena_name = self.np_random.choice(
            ["battlefield", "final_destination"],
            p=[0.8, 0.2],
        )

        self.arena.load(arena_name)
        self.match.start()

        if self.renderer:
            self.renderer.set_arena(self.arena)

        observation = get_observation(
            self.fighter,
            self.opponent,
            self.arena,
        )

        return observation, {}

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None

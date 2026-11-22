import numpy as np

import config
import simulation as s
from ai.observations import get_observation
from ai.rewards import calculate_reward
from controllers import AIController

arena = s.Arena(config.ARENA_SCALE)
arena.load("battlefield")

player = s.Fighter(name="Player")
jimmy = s.Fighter(name="Jimmy")

fighters = [player, jimmy]
match = s.Match(fighters, arena, max_time=config.MATCH_MAX_TIME)
match.start()

if config.RENDER:
    from controllers import PlayerController
    from renderer import Renderer

    renderer = Renderer(arena)
    controller = PlayerController(renderer)
    ai = AIController()

    arena.load("battlefield")
    match.start()

    fighter = match.fighters[0]
    opponent = match.fighters[1]

    observation = get_observation(fighter, opponent, arena)

    print(observation)
    print(observation.shape)
    print(np.isfinite(observation).all())

    while renderer.running:
        dt = renderer.dt
        actions = controller.get_actions()
        ai_actions = ai.get_actions()

        player.apply_input(actions)
        jimmy.apply_input(ai_actions)

        if not renderer.running:
            break

        result = match.tick(dt)

        for fighter in match.fighters:
            reward = calculate_reward(fighter, match.events, dt)

            if reward:
                print(f"Reward: {reward}")

        for event in match.events:
            print(event["type"], event)

        if result is False:
            pass
        else:
            print(f"Winner: {result}")
            break

        renderer.draw(match.fighters, match)

    renderer.close()
else:
    dt = 1 / 60

    while True:
        result = match.tick(dt)

        if result is not False:
            break

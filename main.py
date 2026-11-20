import config
import simulation as s

arena = s.Arena(config.ARENA_SCALE)
arena.load("battlefield")

player = s.Fighter(x=600)
jimmy = s.Fighter(x=200)

fighters = [player, jimmy]
match = s.Match(fighters, arena, config.MATCH_MAX_TIME)

if config.RENDER:
    from controllers import PlayerController
    from renderer import Renderer

    renderer = Renderer(arena)
    controller = PlayerController(renderer)

    while renderer.running:
        dt = renderer.dt
        actions = controller.get_actions()

        player.apply_input(actions)

        if not renderer.running:
            break
        result = match.tick(dt)

        renderer.draw(match.fighters)

    renderer.close()
else:
    dt = 1 / 60

    while True:
        result = match.tick(dt)

        if result is not False:
            break

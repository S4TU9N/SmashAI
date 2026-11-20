class PlayerController:
    def __init__(self, renderer):
        self.renderer = renderer

    def get_actions(self):
        actions = set()
        inputs = self.renderer.get_input()

        for key, pressed in inputs.items():
            if pressed:
                match key:
                    case "left":
                        actions.add("move_left")
                    case "right":
                        actions.add("move_right")
                    case "jump":
                        actions.add("jump")

        return actions

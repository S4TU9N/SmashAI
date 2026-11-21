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
                    case "jab":
                        actions.add("jab")
                    case "heavy":
                        actions.add("heavy")

        return actions


class AIController:
    def get_actions(self):
        return {}

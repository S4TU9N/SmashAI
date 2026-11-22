class PlayerController:
    def __init__(self, renderer):
        self.renderer = renderer
        self.previous_inputs = {}

    def get_actions(self):
        actions = set()
        inputs = self.renderer.get_input()

        for key, pressed in inputs.items():
            if key == "left" and pressed:
                actions.add("move_left")

            elif key == "right" and pressed:
                actions.add("move_right")

            elif key in ("jump", "jab", "heavy"):
                if pressed and not self.previous_inputs.get(key, False):
                    actions.add(key)

        self.previous_inputs = inputs.copy()
        return actions


class AIController:
    def get_actions(self):
        return {}

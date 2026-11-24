class BasicOpponent:
    def get_actions(self, fighter, opponent):
        actions = set()

        distance = opponent.x - fighter.x
        abs_distance = abs(distance)

        if abs_distance > 100:
            actions.add("move_right" if distance > 0 else "move_left")

        if abs_distance < 100:
            actions.add("jab")

        if fighter.grounded and abs_distance < 200:
            actions.add("jump")

        return actions

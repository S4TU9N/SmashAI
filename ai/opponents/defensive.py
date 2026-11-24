class DefensiveOpponent:
    def get_actions(self, fighter, opponent, arena):
        actions = set()

        distance = opponent.x - fighter.x
        abs_distance = abs(distance)

        platform = self.get_platform(fighter, arena)

        if platform:
            center_x = fighter.x + fighter.width / 2
            left_distance = center_x - platform.left
            right_distance = platform.right - center_x

            if left_distance < 70 and fighter.can_jump():
                actions.update({"move_right", "jump"})
                return actions

            if right_distance < 70 and fighter.can_jump():
                actions.update({"move_left", "jump"})
                return actions

        if abs_distance < 120:
            actions.add("move_left" if distance > 0 else "move_right")

        elif abs_distance > 180:
            actions.add("move_right" if distance > 0 else "move_left")

        return actions

    def get_platform(self, fighter, arena):
        for platform in arena.platforms:
            if (
                abs(fighter.bottom - platform.top) < 1
                and fighter.right > platform.left
                and fighter.left < platform.right
            ):
                return platform

        return None

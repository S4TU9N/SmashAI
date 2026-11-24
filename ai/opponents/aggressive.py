class AggressiveOpponent:
    def __init__(self):
        self.attack_cooldown = 0

    def get_actions(self, fighter, opponent):
        actions = set()

        distance = opponent.x - fighter.x
        abs_distance = abs(distance)

        if abs_distance > 90:
            actions.add("move_right" if distance > 0 else "move_left")

        if fighter.grounded and abs_distance < 180:
            actions.add("jump")

        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        elif fighter.attack is None and fighter.hitstun_timer <= 0:
            if abs_distance < 75:
                actions.add("jab")
                self.attack_cooldown = 15

            elif abs_distance < 160:
                actions.add("heavy")
                self.attack_cooldown = 35

        return actions

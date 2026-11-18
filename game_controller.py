
"""Phase 11: Send gesture actions to Subway Surfers through keyboard input."""

import time
import pydirectinput


class GameController:
    """Maps gesture actions to BlueStacks keyboard controls."""

    KEY_MAP = {
        "LEFT": "left",
        "RIGHT": "right",
        "JUMP": "up",
        "SLIDE": "down",
    }

    def __init__(self, press_duration=0.12, cooldown=0.20):
        self.press_duration = press_duration
        self.cooldown = cooldown
        self.last_action = None
        self.last_press_time = 0.0

        # Avoid unnecessary delays between simulated key operations.
        pydirectinput.PAUSE = 0

    def send_action(self, action):
        """Send one keyboard press for a recognized gesture."""

        action = (action or "NONE").upper()

        if action == "NONE":
            self.last_action = None
            return False

        if action not in self.KEY_MAP:
            raise ValueError(f"Unknown action: {action}")

        # Prevent the same held gesture from triggering repeatedly.
        if action == self.last_action:
            return False

        current_time = time.monotonic()

        # Prevent commands from being sent too frequently.
        if current_time - self.last_press_time < self.cooldown:
            return False

        key = self.KEY_MAP[action]

        pydirectinput.keyDown(key)
        try:
            time.sleep(self.press_duration)
        finally:
            pydirectinput.keyUp(key)

        self.last_action = action
        self.last_press_time = time.monotonic()

        return True

    def reset(self):
        """Release all mapped keys and reset the controller."""

        for key in self.KEY_MAP.values():
            pydirectinput.keyUp(key)

        self.last_action = None

    def close(self):
        """Safely release keys when the controller is closed."""
        self.reset()

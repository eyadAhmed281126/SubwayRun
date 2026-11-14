
import time
from game_controller import GameController

controller = GameController(
    press_duration=0.12,
    cooldown=0.20
)

actions = ["LEFT", "RIGHT", "JUMP", "SLIDE"]

try:
    print("Starting keyboard test...")
    print("Click the Subway Surfers game window now!")

    for remaining in range(5, 0, -1):
        print(f"Starting in {remaining}...")
        time.sleep(1)

    for action in actions:
        print(f"Sending: {action}")

        success = controller.send_action(action)
        print(f"Sent successfully: {success}")

        time.sleep(0.5)
        controller.send_action("NONE")
        time.sleep(0.3)

    print("Keyboard test finished!")

finally:
    controller.close()

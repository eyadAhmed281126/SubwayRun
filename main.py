
import cv2

from camera import Camera
from pose_detector import PoseDetector
from gesture_controller import GestureController
from game_controller import GameController


def main():

    camera = Camera()
    detector = PoseDetector()
    controller = GestureController()
    game_controller = GameController()

    try:
        with camera as cam:

            calibration_message = "Press C to calibrate"

            # c = start calibration
            # r = reset calibration
            # q = close

            while True:

                frame = cam.read()

                annotated_frame, landmarks = detector.process(frame)

                detector.draw_info(annotated_frame, landmarks)

                # Calibration
                if controller.is_calibrating:
                    completed = controller.calibrate(landmarks)

                    current, total = controller.get_calibration_progress()

                    calibration_message = f"Calibrating: {current}/{total}"

                    if completed:
                        calibration_message = "Calibration Complete!"
                        print("Real camera baseline:", controller.baseline)

                # Determine current action
                if controller.is_calibrated and not controller.is_calibrating:
                    action = controller.get_action(landmarks)
                else:
                    action = "NONE"

                # Send gesture action to Subway Surfers
                game_controller.send_action(action)

                # Get frame dimensions
                height, width = annotated_frame.shape[:2]
                center_x = width // 2

                # Draw center divider
                cv2.line(
                    annotated_frame,
                    (center_x, 0),
                    (center_x, height),
                    (255, 255, 255),
                    2
                )

                # Display STATUS
                cv2.putText(
                    annotated_frame,
                    "STATUS",
                    (10, height - 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                # Display current action
                cv2.putText(
                    annotated_frame,
                    action,
                    (10, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                # Display calibration message
                cv2.putText(
                    annotated_frame,
                    calibration_message,
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.imshow("SubwayRun", annotated_frame)

                key = cv2.waitKey(1) & 0xFF

                # Start calibration
                if key == ord('c') and not controller.is_calibrated and not controller.is_calibrating:
                    controller.start_calibration()
                    calibration_message = "Stand still for calibration"

                # Reset calibration
                if key == ord('r'):
                    controller.reset_calibration()
                    game_controller.reset()
                    calibration_message = "Press C to calibrate"

                # Exit
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")

    finally:
        game_controller.close()
        detector.close()
        cv2.destroyAllWindows()
        print("SubwayRun controller stopped safely.")


if __name__ == "__main__":
    main()

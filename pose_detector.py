"""Phases 2 & 3: Pose detection, tracking and skeleton visualization.

Menna (Phase 4+) uses:
    from pose_detector import PoseDetector
    detector = PoseDetector()
    annotated_frame, landmarks = detector.process(frame)

`landmarks` is None if no person is detected, otherwise a list of 33 dicts:
    {"id", "name", "x", "y", "z", "visibility", "px", "py"}
x, y are normalized (0-1), px, py are pixel coordinates, z is relative depth.
"""
import time
import cv2
import mediapipe as mp


class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1,
                 min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._prev_time = time.time()
        self.fps = 0.0

    def process(self, frame, draw=True):
        """Run pose detection on a BGR frame.

        Returns (annotated_frame, landmarks_or_None). Never raises when
        no person is visible.
        """
        output = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)

        landmarks = None
        if results.pose_landmarks:
            landmarks = self._extract(results.pose_landmarks, frame.shape)
            if draw:
                self.mp_drawing.draw_landmarks(
                    output, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=3, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(255, 255, 255), thickness=2),
                )
        return output, landmarks

    def _extract(self, pose_landmarks, shape):
        h, w = shape[:2]
        data = []
        for i, lm in enumerate(pose_landmarks.landmark):
            data.append({
                "id": i,
                "name": self.mp_pose.PoseLandmark(i).name,
                "x": lm.x, "y": lm.y, "z": lm.z,
                "visibility": lm.visibility,
                "px": int(lm.x * w), "py": int(lm.y * h),
            })
        return data

    def draw_info(self, frame, landmarks):
        """Draw FPS and person-detected status on the frame (in place)."""
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if landmarks is None:
            cv2.putText(frame, "No person detected", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    def close(self):
        self.pose.close()


def get_landmarks(frame, detector):
    """Convenience wrapper: returns (annotated_frame, landmarks)."""
    return detector.process(frame)
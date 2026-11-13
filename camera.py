"""Phase 1: Camera handling (open / read frames / release / error handling)."""
import cv2


class CameraError(Exception):
    """Raised when the camera can't be opened or read."""


class Camera:
    def __init__(self, index=0, width=640, height=480, mirror=True):
        self.index = index
        self.width = width
        self.height = height
        self.mirror = mirror  # flip horizontally so left/right feel natural
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            self.cap = None
            raise CameraError(
                f"Cannot open camera index {self.index}. "
                "Check that the webcam is connected and not used by another app."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self

    def read(self):
        """Return one BGR frame. Raises CameraError if reading fails."""
        if self.cap is None:
            raise CameraError("Camera is not open. Call open() first.")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise CameraError("Failed to read frame from camera.")
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    # allows: with Camera() as cam:
    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False
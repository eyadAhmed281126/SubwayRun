import time

# CONSTANTS
# NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


class GestureController:

    def __init__(self):
        self.min_visibility = 0.7      # the least visibility we accept
        self.baseline = None
        self.calibration_frames = 60   # 60 correct features ready to be used
        self.calibration_samples = []  # collected features
        self.is_calibrated = False     # if we finished
        self.is_calibrating = False    # if we are calibrating currently

        # Calibration stability: if the person moves during calibration
        # (std of shoulder position above this), the samples are thrown away
        # and calibration restarts automatically.
        self.calibration_max_std = 0.015
        self.calibration_restarts = 0  # how many times it restarted (for UI/debug)
        self.min_torso_height = 0.05   # sanity check for calibration
        # All thresholds are RATIOS of the baseline torso_height
        # (so they don't depend on the distance from the camera).
        
        self.jump_threshold = -0.25    # shoulder goes UP (y decreases)
        self.slide_threshold = 0.25    # shoulder goes DOWN (y increases)
        self.left_threshold = -0.25
        self.right_threshold = 0.25

        self.jump_cooldown = 0.6       # seconds between two jumps
        self.last_jump_time = 0.0

    # ------------------------------------------------------------------
    # Landmarks helpers
    # ------------------------------------------------------------------
    def is_landmark_visible(self, landmark):
        return landmark["visibility"] >= self.min_visibility

    def _center(self, landmarks, left_idx, right_idx): #To calculate the center in general
        if landmarks is None:
            return None

        left = landmarks[left_idx]
        right = landmarks[right_idx]

        if not self.is_landmark_visible(left) or not self.is_landmark_visible(right):
            return None

        return ((left["x"] + right["x"]) / 2, (left["y"] + right["y"]) / 2)

    def get_shoulder_center(self, landmarks):
        return self._center(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)

    def get_hip_center(self, landmarks):
        return self._center(landmarks, LEFT_HIP, RIGHT_HIP)

    def extract_features(self, landmarks, require_hip=True):

        #require_hip = True  -> used in calibration (we need the full torso).
        #require_hip = False -> used at runtime (shoulders are enough).

        if landmarks is None:
            return None

        shoulder_center = self.get_shoulder_center(landmarks)
        if shoulder_center is None:
            return None

        hip_center = self.get_hip_center(landmarks)
        if require_hip and hip_center is None:
            return None

        features = {"shoulder_center": shoulder_center}

        if hip_center is not None:
            features["hip_center"] = hip_center
            features["torso_height"] = hip_center[1] - shoulder_center[1]

        return features

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def start_calibration(self):  # reset everything to start fresh with a new person
        self.baseline = None
        self.calibration_samples = []
        self.calibration_restarts = 0
        self.is_calibrated = False
        self.is_calibrating = True  # we are ready to accept new samples

    @staticmethod
    def _std(values):
        mean = sum(values) / len(values)
        return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    def _is_stable(self, samples):
        #True if the shoulders stayed (almost) still during calibration.
        xs = [s["shoulder_center"][0] for s in samples]
        ys = [s["shoulder_center"][1] for s in samples]
        return (
            self._std(xs) < self.calibration_max_std #calculates Standard Deviation & make sure it is less than 0.015
            and self._std(ys) < self.calibration_max_std
        )

    def _calculate_baseline(self):  # average of the collected features
        samples = self.calibration_samples
        count = len(samples)

        baseline = {}

        # EXAMPLE
        # {
        #   "shoulder_center": (0.41, 0.4033),
        #   "hip_center": (0.46, 0.7033),
        #   "torso_height": 0.30   <- not a tuple
        # }

        for key in samples[0]:
            if isinstance(samples[0][key], tuple):
                baseline[key] = (
                    sum(s[key][0] for s in samples) / count,  # average x
                    sum(s[key][1] for s in samples) / count,  # average y
                )
            else:
                baseline[key] = sum(s[key] for s in samples) / count

        return baseline

    def calibrate(self, landmarks):
        if not self.is_calibrating:
            return False

        features = self.extract_features(landmarks, require_hip=True)

        if features is None:
            return False

        # ignore bad frames (e.g. person bending or tracking glitch)
        if features["torso_height"] < self.min_torso_height:
            return False

        self.calibration_samples.append(features)

        if len(self.calibration_samples) >= self.calibration_frames:
            if not self._is_stable(self.calibration_samples):
                # person moved while calibrating -> throw away and try again
                self.calibration_samples = []
                self.calibration_restarts += 1
                return False

            self.baseline = self._calculate_baseline()
            self.is_calibrating = False
            self.is_calibrated = True
            return True

        return False

    def reset_calibration(self):  # delete and stop
        self.baseline = None
        self.calibration_samples = []
        self.calibration_restarts = 0
        self.is_calibrated = False
        self.is_calibrating = False

    def get_calibration_progress(self):
        return len(self.calibration_samples), self.calibration_frames

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------
    def get_movement(self, landmarks):
        if not self.is_calibrated:  # no baseline -> nothing to compare with
            return None

        features = self.extract_features(landmarks, require_hip=False)

        #visibility of the hip is not required to identify the movement.

        if features is None:
            return None

        th = self.baseline["torso_height"]  # normalization factor
        #the movement is unaffected by the person’s distance from the camera.
        movement = {
            "shoulder_y": (features["shoulder_center"][1] - self.baseline["shoulder_center"][1]) / th,
            "shoulder_x": (features["shoulder_center"][0] - self.baseline["shoulder_center"][0]) / th,
            "hip_y": None,
            "hip_x": None,
        }

        # hip is optional (might be out of frame)
        if "hip_center" in features:
            movement["hip_y"] = (features["hip_center"][1] - self.baseline["hip_center"][1]) / th
            movement["hip_x"] = (features["hip_center"][0] - self.baseline["hip_center"][0]) / th

        return movement

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------
    def detect_jump(self, movement):
        if movement is None:
            return False

        return movement["shoulder_y"] < self.jump_threshold

    def detect_slide(self, movement):
        if movement is None:
            return False

        return movement["shoulder_y"] > self.slide_threshold

    def detect_left(self, movement):
        if movement is None:
            return False

        return movement["shoulder_x"] < self.left_threshold

    def detect_right(self, movement):
        if movement is None:
            return False

        return movement["shoulder_x"] > self.right_threshold

    # ------------------------------------------------------------------
    # Final action
    # ------------------------------------------------------------------
    def get_action(self, landmarks):  # turn landmarks into actions
        movement = self.get_movement(landmarks)

        if movement is None:
            return "NONE"

        if self.detect_jump(movement):
            now = time.monotonic() #Prevents repeated jump actions within the cooldown interval.
            if now - self.last_jump_time >= self.jump_cooldown: 
                self.last_jump_time = now
                return "JUMP"
            return "NONE"  # still in cooldown, don't fall through to other actions

        if self.detect_slide(movement):
            return "SLIDE"

        # If the camera image is mirrored, verify whether LEFT and RIGHT
        # actions need to be swapped to match the player's real-world movement.
        if self.detect_left(movement):
            return "LEFT"

        if self.detect_right(movement):
            return "RIGHT"

        return "NONE"
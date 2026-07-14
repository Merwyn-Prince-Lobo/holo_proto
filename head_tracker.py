"""
head_tracker.py

Webcam + MediaPipe Face Mesh based head position tracker.

Why this approach instead of literally tracking pupils:
- Pupil tracking is noisy and really answers "where are the eyes looking",
  not "where is the head in 3D space". For parallax/hologram effect we
  need head position, not gaze direction.
- We use the eye-corner landmarks (stable, low-jitter) and estimate the
  inter-pupillary distance (IPD) in pixels. Since real IPD is roughly
  constant (~63mm average adult), the apparent pixel distance shrinking
  or growing tells us how close/far the head is (a poor-man's depth
  estimate from a single 2D camera, no stereo needed).
- Left/right and up/down position comes from where the midpoint between
  the eyes is in the frame, converted to an angle and then to a lateral
  offset using the estimated distance.

This gives us approximate (x, y, z) of the viewer's head, in centimeters,
relative to the webcam. It's not metrically perfect (everyone's IPD is a
bit different) but it's plenty stable and consistent for a parallax demo.
"""

import time
import collections
import cv2
import numpy as np
import mediapipe as mp


class HeadPose:
    """Simple container for a smoothed head position estimate."""
    __slots__ = ("x", "y", "z", "timestamp")

    def __init__(self, x=0.0, y=0.0, z=60.0, timestamp=None):
        self.x = x          # cm, +right
        self.y = y          # cm, +up
        self.z = z          # cm, distance from camera (along camera axis)
        self.timestamp = timestamp or time.time()

    def __repr__(self):
        return f"HeadPose(x={self.x:.1f}, y={self.y:.1f}, z={self.z:.1f})"


class EMAFilter:
    """Exponential moving average filter, one instance per scalar value."""

    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value


class HeadTracker:
    """
    Wraps webcam capture + MediaPipe Face Mesh to produce a smoothed,
    real-time HeadPose estimate.

    Usage:
        tracker = HeadTracker()
        tracker.start()
        while True:
            pose = tracker.get_pose()   # non-blocking, returns latest estimate
            if pose is None:
                continue  # no face detected yet
            ...
        tracker.stop()
    """

    # Average adult interpupillary distance in cm. Used to convert pixel
    # distances into an approximate real-world depth (z).
    ASSUMED_IPD_CM = 6.3

    # Left/right eye outer-corner landmark indices in MediaPipe's 468-point
    # face mesh topology. These are anatomically stable points (corner of
    # the eye socket), less prone to jitter than pupil/iris landmarks.
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    def __init__(self, camera_index=0, frame_width=640, frame_height=480,
                 smoothing_alpha=0.35, focal_length_px=None):
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.cap = None
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Focal length in pixels. If not supplied, we approximate it from
        # frame width assuming a "typical" laptop webcam horizontal FOV of
        # about 60 degrees. This is a rough calibration; see calibrate()
        # for a better way to set this from a known distance.
        if focal_length_px is None:
            assumed_hfov_deg = 60.0
            focal_length_px = (frame_width / 2.0) / np.tan(np.radians(assumed_hfov_deg / 2.0))
        self.focal_length_px = focal_length_px

        self.filter_x = EMAFilter(smoothing_alpha)
        self.filter_y = EMAFilter(smoothing_alpha)
        self.filter_z = EMAFilter(smoothing_alpha)

        self._last_pose = None
        self._fps_window = collections.deque(maxlen=30)
        self._last_frame_time = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {self.camera_index}. "
                "Check that no other app is using it and the index is correct."
            )

    def stop(self):
        if self.cap is not None:
            self.cap.release()
        self.face_mesh.close()

    def calibrate(self, known_distance_cm):
        """
        Optional one-shot calibration: have the user sit at a known
        distance (e.g. measure 50cm with a ruler) and call this. It
        recomputes focal_length_px from the currently observed eye-pixel
        distance, which corrects for the rough 60-degree FOV assumption
        and per-person IPD variance in one go.
        """
        ok, frame = self.cap.read()
        if not ok:
            return False
        eye_px_dist = self._measure_eye_pixel_distance(frame)
        if eye_px_dist is None:
            return False
        # focal_length = (real_size * pixel_size) / real_distance, rearranged
        # from the standard pinhole projection: pixel_size = focal * real_size / distance
        self.focal_length_px = (eye_px_dist * known_distance_cm) / self.ASSUMED_IPD_CM
        return True

    def _measure_eye_pixel_distance(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        left = landmarks[self.LEFT_EYE_OUTER]
        right = landmarks[self.RIGHT_EYE_OUTER]
        lx, ly = left.x * w, left.y * h
        rx, ry = right.x * w, right.y * h
        return float(np.hypot(rx - lx, ry - ly))

    def read_frame_and_landmarks(self):
        """Grabs one frame and runs face mesh. Returns (frame, landmarks_or_None)."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return frame, None
        return frame, results.multi_face_landmarks[0].landmark

    def update(self):
        """
        Reads one frame, estimates head pose, applies smoothing, and
        stores it as the latest pose. Call this once per loop iteration
        in your main render loop (or a dedicated tracking thread).
        Returns the smoothed HeadPose, or None if no face detected.
        """
        frame, landmarks = self.read_frame_and_landmarks()
        if frame is None:
            return self._last_pose
        if landmarks is None:
            # No face found this frame; hold last known pose rather than
            # snapping back to center, so the view doesn't jump around.
            return self._last_pose

        h, w = frame.shape[:2]
        left = landmarks[self.LEFT_EYE_OUTER]
        right = landmarks[self.RIGHT_EYE_OUTER]
        lx, ly = left.x * w, left.y * h
        rx, ry = right.x * w, right.y * h

        eye_px_dist = float(np.hypot(rx - lx, ry - ly))
        eye_px_dist = max(eye_px_dist, 1e-3)  # avoid divide-by-zero

        # Depth (z) from apparent IPD size: closer face -> larger pixel
        # distance between eyes. Standard pinhole inverse relationship.
        z_cm = (self.ASSUMED_IPD_CM * self.focal_length_px) / eye_px_dist

        # Midpoint between eyes in pixel space, relative to frame center.
        mid_x_px = (lx + rx) / 2.0 - (w / 2.0)
        mid_y_px = (ly + ry) / 2.0 - (h / 2.0)

        # Convert pixel offset to a real-world lateral offset at distance z,
        # using similar triangles: offset_cm = (pixel_offset / focal_length) * z
        x_cm = (mid_x_px / self.focal_length_px) * z_cm
        # Image y grows downward; we want +y = up, so flip sign.
        y_cm = -(mid_y_px / self.focal_length_px) * z_cm

        smoothed_x = self.filter_x.update(x_cm)
        smoothed_y = self.filter_y.update(y_cm)
        smoothed_z = self.filter_z.update(z_cm)

        self._last_pose = HeadPose(smoothed_x, smoothed_y, smoothed_z)

        now = time.time()
        if self._last_frame_time is not None:
            self._fps_window.append(1.0 / max(now - self._last_frame_time, 1e-6))
        self._last_frame_time = now

        return self._last_pose

    def get_pose(self):
        return self._last_pose

    @property
    def fps(self):
        if not self._fps_window:
            return 0.0
        return sum(self._fps_window) / len(self._fps_window)

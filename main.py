"""
main.py

Eye/head-tracked "hologram" prototype.

Run with:
    python main.py

Controls:
    ESC / Q   - quit
    C         - calibrate (sit at exactly 50cm from the screen and press C;
                this improves the depth/distance accuracy a lot)
    F         - toggle fullscreen
    SPACE     - toggle a slow auto-rotation of the model (good for showing
                off the parallax effect to someone else, or for demoing
                without a face in frame)

What you should see:
    A 3D model sitting "inside" the screen. As you move your head left,
    right, up, down, closer, or farther, the rendered view updates so the
    object appears to stay fixed in space rather than stuck to the glass --
    like looking through a window into a small diorama.

Notes on getting a convincing effect:
    - Distance matters. Sit at a normal laptop-use distance (40-70cm).
      Too close and tracking gets jittery; too far and MediaPipe loses
      the eye landmarks' precision.
    - Lighting matters for MediaPipe. A well-lit face tracks much better
      than a backlit one.
    - LATERAL_GAIN below exaggerates head movement slightly, since real
      head movement in front of a small laptop screen is subtle. Tune it
      to taste -- higher feels more dramatic but can feel "swimmy."
"""

import sys
import threading
import time

import numpy as np
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    Mat4,
    MatrixLens,
    loadPrcFileData,
    WindowProperties,
    AmbientLight,
    DirectionalLight,
    Vec4,
)

from head_tracker import HeadTracker
from offaxis_projection import Screen, build_panda3d_matrices, head_to_world


# ---- Tunable parameters -----------------------------------------------

# Physical screen size you're rendering on, in cm. Measure your actual
# laptop's visible screen area for the most convincing effect; doesn't
# need to be perfect for the prototype.
SCREEN_WIDTH_CM = 30.0
SCREEN_HEIGHT_CM = 18.0

# Exaggerate lateral head movement so the parallax effect reads clearly
# even with the small natural head movement you make sitting at a laptop.
LATERAL_GAIN = 1.6

# Webcam is usually mounted a couple cm above the screen's vertical
# center and right at the screen's horizontal center on a laptop --
# negligible for this prototype, kept at 0.
DEPTH_OFFSET_CM = 0.0

# How far "into" the screen (away from viewer) the model sits, in cm.
# IMPORTANT: this is in PANDA's native coordinate convention (Z-up,
# Y-forward), used directly for model.setPos() -- NOT the same axes as
# the off-axis math in offaxis_projection.py (which works in an
# OpenGL-style Y-up/Z-toward-viewer frame and only talks to Panda
# through build_panda3d_matrices' internal conversion). Per that
# conversion, "away from the viewer" corresponds to Panda's +Y axis.
# Positive = behind the glass (the classic "diorama" look). You can also
# try a small NEGATIVE value to make the model pop OUT toward the
# viewer instead, for a different but equally valid hologram feel.
MODEL_DEPTH_CM = 25.0

NEAR_CM = 5.0
FAR_CM = 400.0


def to_panda_mat4(np_mat4_panda_convention):
    """
    Converts a NumPy 4x4 array that is ALREADY in Panda3D's row-vector
    convention (i.e. the kind build_panda3d_matrices() returns) into a
    Panda3D Mat4. Panda's Mat4(*16 values) constructor takes values in
    row-major order matching its own internal row-vector layout, so this
    is a direct flatten -- no further transpose needed here (the
    transpose work already happened inside build_panda3d_matrices).
    """
    return Mat4(*np_mat4_panda_convention.flatten().tolist())


class HoloApp(ShowBase):
    def __init__(self):
        loadPrcFileData("", "win-size 1280 800")
        loadPrcFileData("", "window-title Eye-Tracked Hologram Prototype")
        ShowBase.__init__(self)

        self.disableMouse()
        self.setBackgroundColor(0.04, 0.04, 0.06)

        self.screen = Screen(SCREEN_WIDTH_CM, SCREEN_HEIGHT_CM)

        # --- Load the model ---------------------------------------------
        # Prototype default: Panda3D's built-in sample model so this runs
        # with zero extra downloads. Swap self.model for loader.loadModel("yourfile.glb")
        # once you've confirmed the parallax effect looks right.
        self.model = self.loader.loadModel("models/panda")
        self.model.reparentTo(self.render)
        self.model.setScale(0.6, 0.6, 0.6)
        self.model.setPos(0, MODEL_DEPTH_CM, 0)
        self.model.setHpr(180, 0, 0)

        self._setup_lighting()

        # --- Camera lens setup -------------------------------------------
        # The default camera lens is a PerspectiveLens, which only supports
        # fov/aspect-ratio style parameters -- it has no setUserMat(). To
        # drive the camera with a fully custom (asymmetric/off-axis)
        # projection matrix every frame, we need a MatrixLens instead,
        # which exists specifically for handing Panda a raw projection
        # matrix. We swap it onto the existing camera node.
        self.matrix_lens = MatrixLens()
        self.cam.node().setLens(self.matrix_lens)

        # --- Head tracking (runs in a background thread so webcam I/O
        # and MediaPipe inference never block the render loop) -----------
        self.tracker = HeadTracker(camera_index=0)
        self.tracker.start()
        self._tracker_lock = threading.Lock()
        self._latest_pose = None
        self._tracking_thread_stop = False
        self._tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._tracking_thread.start()

        self._auto_rotate = False
        self._rotate_heading = 180.0

        # --- Input ---------------------------------------------------
        self.accept("escape", sys.exit)
        self.accept("q", sys.exit)
        self.accept("c", self._calibrate)
        self.accept("f", self._toggle_fullscreen)
        self.accept("space", self._toggle_autorotate)

        self.taskMgr.add(self._update_task, "update-task")

        # On-screen debug readout.
        self._debug_text = self._make_debug_text()

    def _setup_lighting(self):
        amb = AmbientLight("amb")
        amb.setColor(Vec4(0.4, 0.4, 0.42, 1))
        amb_np = self.render.attachNewNode(amb)
        self.render.setLight(amb_np)

        dlight = DirectionalLight("dlight")
        dlight.setColor(Vec4(0.9, 0.9, 0.85, 1))
        dlight_np = self.render.attachNewNode(dlight)
        dlight_np.setHpr(45, -45, 0)
        self.render.setLight(dlight_np)

    def _make_debug_text(self):
        from direct.gui.OnscreenText import OnscreenText
        return OnscreenText(
            text="Initializing webcam + face tracking...",
            pos=(-1.3, 0.92),
            scale=0.05,
            fg=(0.8, 1.0, 0.8, 1),
            align=0,
            mayChange=True,
        )

    def _tracking_loop(self):
        """Runs on a background thread: continuously update head pose."""
        while not self._tracking_thread_stop:
            pose = self.tracker.update()
            with self._tracker_lock:
                self._latest_pose = pose
            time.sleep(0.001)  # yield briefly; webcam read is the real pacing limit

    def _calibrate(self):
        ok = self.tracker.calibrate(known_distance_cm=50.0)
        msg = "Calibrated at 50cm." if ok else "Calibration failed (no face detected)."
        print(msg)

    def _toggle_fullscreen(self):
        props = WindowProperties()
        wp = self.win.getProperties()
        props.setFullscreen(not wp.getFullscreen())
        self.win.requestProperties(props)

    def _toggle_autorotate(self):
        self._auto_rotate = not self._auto_rotate

    def _update_task(self, task):
        with self._tracker_lock:
            pose = self._latest_pose

        if self._auto_rotate:
            self._rotate_heading += 20.0 * globalClock.getDt()
            self.model.setHpr(self._rotate_heading, 0, 0)

        if pose is None:
            self._debug_text.setText("No face detected -- move into webcam view.")
            return Task.cont

        eye_world = head_to_world(
            pose,
            lateral_gain=LATERAL_GAIN,
            depth_offset_cm=DEPTH_OFFSET_CM,
        )

        cam_transform_np, projection_np = build_panda3d_matrices(
            eye_world, self.screen, near=NEAR_CM, far=FAR_CM
        )

        self.matrix_lens.setUserMat(to_panda_mat4(projection_np))
        self.camera.setMat(to_panda_mat4(cam_transform_np))

        self._debug_text.setText(
            f"head: x={pose.x:+5.1f}cm y={pose.y:+5.1f}cm z={pose.z:5.1f}cm   "
            f"tracker fps: {self.tracker.fps:4.1f}   "
            f"[C]alibrate  [F]ullscreen  [SPACE] auto-rotate  [ESC] quit"
        )

        return Task.cont

    def userExit(self):
        self._tracking_thread_stop = True
        self.tracker.stop()
        sys.exit()


if __name__ == "__main__":
    app = HoloApp()
    try:
        app.run()
    except KeyboardInterrupt:
        app._tracking_thread_stop = True
        app.tracker.stop()

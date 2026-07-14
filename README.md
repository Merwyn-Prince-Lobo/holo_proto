# Eye-Tracked Hologram Prototype

A head-tracked "fish tank VR" demo: move your head in front of your
webcam, and a 3D model on screen reacts with real motion parallax, so it
feels like it's sitting behind (or popping out of) the glass instead of
being flat on the screen.

## How it works (short version)

1. `head_tracker.py` -- webcam + MediaPipe Face Mesh estimate your head's
   (x, y, z) position relative to the screen, smoothed with an
   exponential moving average so it doesn't jitter.
2. `offaxis_projection.py` -- the actual "hologram" trick. Instead of
   rotating a camera to look at a target (which makes the *scene* seem
   to spin to face you -- wrong), it builds an **off-axis / asymmetric
   perspective frustum**: the screen is treated as a fixed window into
   the 3D world, and only the eye position moves. This is the same
   technique used in CAVE VR systems and Johnny Lee's classic Wii
   head-tracking demo. The math is verified numerically: the screen
   rectangle's four corners always map exactly to the four corners of
   the viewport, no matter where your head moves.
3. `main.py` -- wires it together in Panda3D, feeding a custom
   projection matrix into a `MatrixLens` and positioning the camera
   every frame based on your tracked head position.

## Setup (Windows)

```
pip install -r requirements.txt
python main.py
```

You'll need a working webcam (built-in laptop cam is fine) and a bit of
decent lighting on your face for MediaPipe to track reliably.

## Controls

- `ESC` or `Q` -- quit
- `C` -- calibrate: sit at exactly 50cm from the screen (use a ruler/tape
  measure once) and press C. This fixes the depth-estimation accuracy,
  which otherwise relies on an assumed "average" interpupillary distance
  and a rough guess at your webcam's field of view.
- `F` -- toggle fullscreen
- `SPACE` -- toggle a slow auto-rotation of the model (useful for
  demoing to someone else, or for checking the model looks right when
  no face is in frame)

## What you should see

A panda model (Panda3D's built-in sample model, used here just so the
prototype runs with zero extra downloads) sitting a bit "behind" the
screen. Move your head left/right/up/down/closer/farther and the view
should update so the model feels anchored in space -- like looking
through a window -- rather than feeling like the camera is just orbiting
around it.

## Tuning

All the knobs are at the top of `main.py`:

- `SCREEN_WIDTH_CM` / `SCREEN_HEIGHT_CM` -- measure your actual laptop's
  visible screen area for the most accurate effect.
- `LATERAL_GAIN` -- exaggerates head movement. Real head movement sitting
  at a laptop is subtle, so this defaults to 1.6x. Try values from 1.0
  (literal/subtle) to ~2.5 (dramatic, can feel "swimmy" if too high).
- `MODEL_DEPTH_CM` -- how far behind the glass the model sits. Try a
  negative value to make it pop out toward you instead.

## Swapping in your own 3D model

Replace this block in `main.py`:

```python
self.model = self.loader.loadModel("models/panda")
```

with:

```python
self.model = self.loader.loadModel("your_model.glb")
```

Panda3D natively supports `.egg`/`.bam` well; for `.glb`/`.gltf` you may
need the `panda3d-gltf` package (`pip install panda3d-gltf`). `.obj` and
`.fbx` typically need conversion to one of Panda's native formats first
(tools like `obj2egg`, or exporting from Blender with a Panda3D egg
exporter). If you hit a "couldn't load model" error, that's almost
always a format/converter issue, not a bug in this tracking pipeline --
shout and I'll help sort the conversion.

## Known limitations of this prototype (by design, since it's a first pass)

- Depth (z) is estimated from interpupillary distance using an assumed
  average IPD of 6.3cm -- not from real stereo depth. It's stable and
  consistent for one person across a session (especially after
  calibrating), but not metrically precise, and differs slightly
  between people with different actual IPDs.
- Single face only (`max_num_faces=1`); if multiple people are in frame,
  MediaPipe will track whichever one it locks onto first.
- No gesture controls, multi-depth-layer 2D image support, or AI
  monocular depth estimation yet -- those are natural "next" features
  once this core parallax loop feels good. Happy to build any of those
  next.

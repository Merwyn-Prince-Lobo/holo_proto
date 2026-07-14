"""
offaxis_projection.py

The core trick that makes this look like a "hologram" instead of just a
camera spinning around an object.

THE KEY IDEA
------------
Your screen is a fixed rectangle floating in real 3D space (in front of
or around the laptop). The 3D scene lives "behind" that rectangle, like
looking through a window. As your head moves, the window (screen) does
NOT move -- but your eye does, so the window onto the world should
behave exactly like a real window would: looking through it from a
different angle reveals different things, and parallax (near objects
shift more than far objects) emerges naturally.

A normal "look-at" camera (camera rotates to point at a target) gets
this wrong: it makes the *scene* seem to rotate to face you, when really
it's your viewpoint *through a fixed frame* that should be changing.

The correct technique is an ASYMMETRIC (off-axis) PERSPECTIVE FRUSTUM,
the same approach used in CAVE VR installations and Wii head-tracking
demos (Johnny Lee's famous setup). Reference: Robert Kooima's paper
"Generalized Perspective Projection" (2009) is the standard write-up of
this exact technique.

HOW IT WORKS
------------
Define the screen as a rectangle in world space with three corners:
    pa = bottom-left corner
    pb = bottom-right corner
    pc = top-left corner
(the fourth corner, top-right, is implied: pb + pc - pa)

Given the eye position `pe`, we:
1. Compute the screen's local basis vectors:
     vr (right)  = (pb - pa) / |pb - pa|
     vu (up)     = (pc - pa) / |pc - pa|
     vn (normal) = vr x vu   (points OUT of the screen toward the eye)
2. Compute vectors from the eye to each screen corner.
3. Project those vectors onto the screen's basis to find the distance
   from the eye to the screen plane (d), and the off-axis left/right/
   top/bottom extents at the near clipping plane.
4. Scale those extents by (near / d) to get the actual frustum bounds at
   the near plane -- this is what makes it "off-axis": if the eye is not
   centered in front of the screen, left/right (and top/bottom) extents
   are asymmetric.
5. Build a standard OpenGL-style asymmetric frustum projection matrix
   from those bounds.
6. Build a view matrix that orients the camera using the *screen's*
   basis vectors (not a look-at-target rotation!) -- the camera looks
   straight along -vn at all times, only the frustum skews.

This file is render-engine agnostic (pure NumPy), so it can be reused
whether the final render uses Panda3D, PyOpenGL, etc.
"""

import numpy as np


class Screen:
    """
    Defines the physical screen as a rectangle in world space.
    Units should match whatever you use for head position (we use cm).

    By convention:
        - World origin (0,0,0) is the center of the screen.
        - +x is to the right (from the user's point of view facing it).
        - +y is up.
        - +z points OUT of the screen toward the user.
    This means the screen plane is z = 0, and the 3D scene "inside" the
    hologram should be placed at z < 0 (behind the glass) and/or
    z > 0 (popping out toward the viewer) for the effect to read well.
    """

    def __init__(self, width_cm, height_cm):
        self.width = width_cm
        self.height = height_cm
        hw, hh = width_cm / 2.0, height_cm / 2.0
        # Corners, counter-clockwise starting bottom-left, in world space.
        self.pa = np.array([-hw, -hh, 0.0])  # bottom-left
        self.pb = np.array([hw, -hh, 0.0])   # bottom-right
        self.pc = np.array([-hw, hh, 0.0])   # top-left


def build_offaxis_view_and_projection(eye_pos, screen: Screen, near=1.0, far=500.0):
    """
    Computes (view_matrix, projection_matrix) as 4x4 NumPy arrays
    (column-major, OpenGL convention: v' = P * V * v) for an asymmetric
    frustum given the eye position relative to a fixed screen.

    Parameters
    ----------
    eye_pos : array-like, shape (3,)
        Eye/head position in the same world-space coordinates as the
        screen (cm), e.g. coming from HeadTracker, after mapping camera
        coordinates into this world frame (see head_to_world below).
    screen : Screen
    near, far : clipping plane distances (cm). `near` should be smaller
        than the eye's expected distance to the screen plane.

    Returns
    -------
    view : 4x4 np.ndarray
    proj : 4x4 np.ndarray
    """
    pe = np.asarray(eye_pos, dtype=np.float64)
    pa, pb, pc = screen.pa, screen.pb, screen.pc

    # Screen basis vectors.
    vr = (pb - pa)
    vr = vr / np.linalg.norm(vr)
    vu = (pc - pa)
    vu = vu / np.linalg.norm(vu)
    vn = np.cross(vr, vu)
    vn = vn / np.linalg.norm(vn)

    # Vectors from eye to screen corners.
    va = pa - pe
    vb = pb - pe
    vc = pc - pe

    # Distance from eye to screen plane (projection onto normal).
    d = -np.dot(va, vn)
    d = max(d, 1e-4)  # guard against eye being on/behind the screen plane

    # Frustum extents at the NEAR plane, found by projecting the corner
    # vectors onto the screen basis and scaling by near/d (similar
    # triangles: the near plane is `near` away, the screen is `d` away).
    scale = near / d
    left = np.dot(vr, va) * scale
    right = np.dot(vr, vb) * scale
    bottom = np.dot(vu, va) * scale
    top = np.dot(vu, vc) * scale

    proj = _asymmetric_frustum(left, right, bottom, top, near, far)

    # View matrix: orient using the screen's own basis (camera looks
    # along -vn), translate by -eye position. This is the "M" matrix
    # from Kooima's paper, then inverted/transposed into a standard
    # look rotation + translation.
    rot = np.array([
        [vr[0], vr[1], vr[2]],
        [vu[0], vu[1], vu[2]],
        [vn[0], vn[1], vn[2]],
    ])
    view = np.eye(4)
    view[:3, :3] = rot
    view[:3, 3] = -rot @ pe

    return view, proj


def _asymmetric_frustum(left, right, bottom, top, near, far):
    """Standard OpenGL-style asymmetric (off-axis) perspective projection."""
    proj = np.zeros((4, 4))
    proj[0, 0] = (2.0 * near) / (right - left)
    proj[0, 2] = (right + left) / (right - left)
    proj[1, 1] = (2.0 * near) / (top - bottom)
    proj[1, 2] = (top + bottom) / (top - bottom)
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -(2.0 * far * near) / (far - near)
    proj[3, 2] = -1.0
    return proj


def panda3d_axis_convention_matrix():
    """
    Our off-axis math above is derived in a generic right-handed
    "OpenGL-style" frame: +x right, +y up, +z toward the viewer (out of
    the screen), camera looks down -z. This is the conventional frame
    for Kooima-style off-axis derivations and most graphics literature.

    Panda3D's default frame is different: +x right, +y forward (into the
    screen, away from viewer), +z up (right-handed, Z-up). Reference:
    Panda3D uses x:right, y:forward, z:up as its default world and view
    space convention.

    Rather than re-deriving the frustum math in Panda's convention (easy
    to introduce sign errors), we keep offaxis_projection.py in the
    standard OpenGL frame and apply a single fixed basis-change matrix
    when handing matrices to Panda3D. This matrix maps:
        our +x (right)          -> panda +x (right)
        our +y (up)              -> panda +z (up)
        our +z (toward viewer)   -> panda -y (since panda +y is INTO the
                                    screen, away from viewer)
    This is a pure axis permutation/reflection (determinant +1, a valid
    rotation -- it does not mirror the scene).
    """
    return np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)


def build_panda3d_matrices(eye_pos_ours, screen: Screen, near=1.0, far=500.0):
    """
    High-level convenience wrapper: given an eye position in OUR
    (OpenGL-style: +x right, +y up, +z toward viewer) coordinate
    convention, and a Screen also defined in that convention, returns
    the (cam_transform, projection) matrices ready to hand directly to
    Panda3D -- i.e. already converted into Panda's row-vector,
    Z-up/Y-forward convention.

    Usage in Panda3D:
        cam_mat, proj_mat = build_panda3d_matrices(eye_world, screen, near, far)
        matrix_lens.setUserMat(Mat4(*proj_mat.flatten().tolist()))
        camera_nodepath.setMat(Mat4(*cam_mat.flatten().tolist()))

    Derivation (verified numerically -- screen corners map to exact NDC
    corners regardless of eye position):
        Let B = panda3d_axis_convention_matrix(): maps a point expressed
            in OUR coords to the same physical point expressed in
            PANDA's coords (pure rotation, B^-1 == B^T).
        Let (V, P) = build_offaxis_view_and_projection(...) in OUR coords,
            using OpenGL's column-vector convention (v' = M @ v).
        The matrix that maps a PANDA-world-space point straight to OUR
        eye-space is:           V_pw = V @ B^T
        Panda uses row-vector convention (v' = v @ M) and wants the
        CAMERA's world transform (not its inverse) for NodePath.setMat(),
        and a projection matrix in the same row-vector convention for
        MatrixLens.setUserMat(). Converting column-vector matrices A
        (s.t. v'_col = A @ v_col) into the equivalent row-vector form is
        v'_row = v_row @ A^T. Combining this with needing the *inverse*
        for the camera transform (Panda wants where the camera IS, our
        V_pw maps world->eye i.e. is the inverse of that) gives:
            cam_transform_panda = inverse(V_pw)^T
            projection_panda    = P^T
    """
    pe_ours = np.asarray(eye_pos_ours, dtype=np.float64)
    view_ours, proj_ours = build_offaxis_view_and_projection(pe_ours, screen, near, far)

    B = panda3d_axis_convention_matrix()
    V_pw = view_ours @ B.T  # maps panda-world-space point -> our eye space

    cam_transform_panda = np.linalg.inv(V_pw).T
    projection_panda = proj_ours.T

    return cam_transform_panda, projection_panda


def head_to_world(pose, lateral_gain=1.0, depth_offset_cm=0.0):
    """
    Maps a HeadTracker.HeadPose (camera-relative cm) into the Screen's
    world coordinate frame defined above (origin = screen center,
    +z = toward viewer).

    pose.x/pose.y from the tracker are already roughly "lateral offset
    from camera's optical axis," which -- if the webcam is mounted at
    the top-center of the screen, as it is on basically every laptop --
    is approximately the same as lateral offset from screen center. We
    apply a small gain factor since the effect typically feels more
    convincing exaggerated a bit (real head movement in front of a
    small laptop screen is subtle).

    depth_offset_cm lets you correct for the webcam-to-screen-center
    distance (usually just a couple cm, often negligible).
    """
    x = pose.x * lateral_gain
    y = pose.y * lateral_gain
    z = pose.z + depth_offset_cm
    return np.array([x, y, z])

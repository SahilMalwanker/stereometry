# Calibration Protocol

## Coordinate Contract

Python uses column-vector projection and zero-based pixels. Intrinsics contain
focal lengths in pixels and a principal point in the original image. Distortion
is `[k1,k2,p1,p2,k3]`. Extrinsics map left coordinates into the right camera:
`X_right = R @ X_left + t`. Thus a parallel rig with its right camera 63 mm to the
right has `t = [-63,0,0]`, not positive 63. Baseline is `norm(t)`.

Sparse triangulation takes raw distorted pixels, undistorts once to normalized
rays and uses `[I|0]` and `[R|t]`. Reprojection checks go back to raw pixels.
Dense reconstruction rectifies both views and uses the resulting Q matrix.
Its XYZ frame is the rectified left-camera frame, not necessarily the original one.

## Capture Gates

- Use a rigid, flat board with a measured square pitch and known print accuracy.
- Disable changing focus, stabilization and resolution. Recalibrate after rig motion.
- Cover image boundaries and vary tilt, orientation and distance. Repeated frontal
  views do not adequately constrain focal length and distortion.
- Use matched exposures. Splitting one camera frame avoids host read-time skew,
  but does not certify the device's internal synchronization.
- Inspect corner ordering. A symmetric board may be decoded with inconsistent
  orientation; paired detection alone cannot prove identity of every corner.
- Hold out a physical dimension not used in fitting. Repeat captures and selections
  to characterize repeatability; do not use training RMS as a metric error bar.

## Hardware Context

The original hardware sessions used a 63 mm-baseline side-by-side stereo camera.
Their raw calibration sequences are not part of this repository, so no historical
millimeter-accuracy result is reproduced or certified here, and all bundled
benchmark parameters are explicitly synthetic.

## Validation Boundaries

Tests establish algebraic correctness and behavior on controlled synthetic inputs.
Calibration recovery is tested on projected corner observations with 0.08 px
Gaussian noise. Image-level calibration is tested on rendered, anti-aliased
checkerboard pairs without lens distortion, including rejection of a pair that
shows no board. Dense matching is tested on three textured integer-shift planes.
These are deliberately different fixtures. None measures generalization to
real sensors. Reverse consistency removes many bad matches, but repetitive texture
can yield mutually consistent wrong disparities.

JSON artifacts carry millimeter units and reject unsupported schemas. NPZ files
include the validity mask; consumers must use it. Invalid XYZ is NaN, while raw
disparity remains available for diagnosis. PLY contains only valid subsampled points
and declares meter units in its header.
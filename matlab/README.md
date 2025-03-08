# MATLAB Hardware Companion

Requires MATLAB R2022a+, Computer Vision Toolbox, Image Processing Toolbox and the
USB Webcam support package. These functions have not been executed here: no MATLAB
runtime or camera is available. They are separate from the tested Python package.

```matlab
addpath('matlab');
camera = sm_open('ZED','2560x720');
sm_preview(camera);
sm_live_target(camera,100);
sm_capture_sequence(camera,'captures',20);
[mono,stereo] = sm_calibrate('captures',12);
[left,right] = sm_grab(camera);
sm_mono_measure(left,mono,12);
sm_stereo_measure(left,right,stereo,'PlyFile','scene.ply');
clear camera
```

Replace 12 with the measured square size in millimeters; never rely on a nominal
print size. Confirm that the first half of the device frame really is the physical
left view.

| Original Lab 3 entry point | Adapted implementation |
| --- | --- |
| `ZedInit` | `sm_open`: explicit device and supported resolution |
| `ZedAcquire` | `sm_grab`: one snapshot, validated even width, split pair |
| basic acquisition | `sm_preview` |
| live target detection | `sm_live_target`: bounded loop and close-to-stop |
| sequence acquisition | `sm_capture_sequence`: paired detection, matching names, overwrite protection |
| single-camera calibration/measurement | `sm_calibrate` and `sm_mono_measure` |
| stereo measurements | `sm_calibrate` and `sm_stereo_measure`: epipolar lines, anaglyphs, rectification, sparse/dense triangulation, PLY |

The MATLAB dense path exposes disparity validity and positive depth, but does not
implement Python's additional reverse-match or texture gates. Do not equate their
masks or quality. MATLAB calibration MAT objects are not interchangeable with the
Python JSON model. Use the captured paired PNGs to calibrate in either environment.
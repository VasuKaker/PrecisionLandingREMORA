# PrecisionLandingREMORA

ROS 2 visual precision landing using AprilTag/Landmark detection, planar homography, calibrated solvePnP pose estimation, and MAVROS companion-side landing controllers. This is the current companion-code snapshot from October 20, 2026.

## Components

- `ros2/src/drone_landing_service/drone_landing_service/lightweight_landing_detector.py`: full-resolution image search with a tracked ROI, tag-corner geometry, planar IPPE/iterative solvePnP, reprojection checks, and metric target offsets.
- `landmark_tag_detector.py`: native AprilTag adapter for the `tagLandmark36h11` family.
- `target_detector.py`: ROS image/topic integration. It still inherits shared implementation from `target_detector_deprecated.py`; that base module remains a required dependency despite its historical name.
- `square_detector_experimental.py`: optional square-board fallback and geometry helpers.
- `precision_landing_controller*.py`: automatic, manual-search, and GPS-assisted ROS landing controllers using standard MAVROS services and setpoints.
- Camera/GPS topic adapters, control arbitration, example camera/controller parameters, and synthetic geometry tests.

The active detector geometry uses recursive Landmark IDs 0–3 with printed footprints of 12.8, 32, 80, and 200 mm. Detected coding-boundary corners span 60% of each printed footprint, and the target outline is 200 mm square. The active standard-tag layout is empty in this snapshot. These are fiducial tags, rather than a general-purpose QR text decoder.

## Dependencies

The captured runtime uses ROS 2 Jazzy, Python 3.12, OpenCV 4.6.0 with `cv2.aruco`, and NumPy 1.26.4. ROS dependencies are listed in `package.xml`.

The native `apriltag` Python extension must include `tagLandmark36h11`. A binding that supports only the standard 36h11 family cannot detect the active recursive target. The native extension and third-party detector implementation are external dependencies and are not bundled here; the included adapter imports that extension. Synthetic solvePnP geometry tests do not require decoding real tag images.

Provide an existing ROS image stream on `/landing/image_mono` and a MAVROS connection separately. Set the camera intrinsics, dimensions, and physical target layout for your camera and printed target.

## Build

```bash
cd ros2
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select drone_landing_service
source install/setup.bash
```

Run image perception:

```bash
ros2 launch drone_landing_service precision_landing.launch.py
```

The public launch defaults to detector-only operation. To also launch the existing ROS manual-search controller and command arbiter:

```bash
ros2 launch drone_landing_service precision_landing.launch.py enable_controller:=true control:=manual
```

Available controller choices are `manual`, `automatic`, and `gps`. Controllers retain `start_on_launch:=false`. The manual-search controller exposes `/landing/start` and `/landing/begin_landing`. GPS example coordinates are `0.0`; replace them with your intended target before using GPS-assisted operation. Landing pose topics are `/landing/target_metric`, `/landing/target_range_m`, and `/landing/target_detected`.

## Tests

With ROS/OpenCV dependencies sourced:

```bash
PYTHONPATH=ros2/src/drone_landing_service:$PYTHONPATH \
  python3 -m pytest -q ros2/src/drone_landing_service/test
```

The active six tests cover synthetic planar pose recovery, recursive-tag scale, and corner order. All six passed on the captured ROS/OpenCV runtime. Three older tests referenced removed multipass/contrast/model-completion methods and failed in the original current snapshot; their source is preserved as `legacy_tests/target_detector_snapshot.py` outside active pytest discovery. Tests do not start flight nodes or validate a complete aircraft deployment.

## Snapshot scope

This repository contains the ROS vision and companion landing code only. It contains no PX4 firmware source, custom-firmware bridge/dialect implementation, takeoff implementation, flight recordings, or private repository history.

Publication changes are limited to removing the firmware-specific contact publisher from the shared detector base, removing unrelated flight-utility entry points, providing a focused public launch file, replacing deployment GPS coordinates with examples, and separating obsolete tests from active test discovery. The detection/PnP and ROS landing-controller algorithms retain their captured source. `SOURCE_MANIFEST.json` records hashes of the published files. Existing package attribution and license metadata are preserved in `package.xml` and `setup.py`.

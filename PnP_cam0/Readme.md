# Monocular Camera Pose Estimation using AprilTag Detection

Robust camera pose estimation from AprilTag grid observations using PnP with RANSAC.

## Overview

Estimates camera poses by detecting a planar AprilTag calibration grid and solving the Perspective-n-Point problem. Each detection provides the 6-DOF transformation between the camera and calibration target.

## Results

**Dataset:** 1,411 frames (20 Hz)

**Detection Performance:**
- Successful poses: 273/1,411 (19.3%)
- Mean reprojection error: 0.687 px
- Sub-pixel accuracy: 96.7% (264/273 frames < 1.0 px)

**Spatial Coverage:**
- Camera path length: 35.0 m

## Method

1. **AprilTag Detection:** aprilgrid library detects 6×9 tag grid
2. **Quality Filtering:** Minimum 10 tags required per frame
3. **Pose Estimation:** cv2.solvePnPRansac with 8px threshold, 0.99 confidence
4. **Refinement:** Sub-pixel corner refinement (2×2 window)

## Implementation Details

**Detection Pipeline:**
- Tag family: tag36h11 (54 tags total)
- Sub-pixel refinement: cornerSubPix with 30 iterations
- Boundary check: 5px margin from image edges
- RANSAC inlier ratio: ~98% average

**Coordinate Frames:**
- Target frame: Origin at tag 0 (bottom-right), Z-axis out of plane
- Camera frame: Standard OpenCV convention
- Output: T_cam_to_target (camera trajectory in target frame)

## Key Findings

- RANSAC consensus requires 10+ tags (40+ corner correspondences)
- Frames with 1-2 tags produce unstable pose estimates
- aprilgrid library avoids segmentation faults on cluttered backgrounds
- Corner detection order: bottom-right, top-right, top-left, bottom-left

## Configuration

**Camera Intrinsics:**
- Resolution: 1280×1024
- Focal length: fx=1469.1, fy=1471.6 px
- Principal point: cx=630.6, cy=526.9 px
- Radial distortion: k1=-0.227, k2=0.172

**AprilTag Grid:**
- Layout: 6 columns × 9 rows
- Tag size: 38 mm
- Tag spacing: 27% (12.96 mm between tags)
- Total grid size: ~290mm × 435mm

## Files
```
output/
├── cam0_trajectory_apriltag.json    # 273 pose estimates
├── cam0_detection.avi               # Detection video
├── detection_analysis.png           # Tag detection statistics  
├── reproj_error_dist.png            # Reprojection error histogram
├── trajectory_and_error.png         # 3D trajectory + error plot
├── trajectory_3d.html               # Interactive 3D viewer
└── reprojection_debug/              # Worst-case error visualizations
```

## Visualization Outputs

**Detection Analysis:** Tag counts per frame, temporal consistency  
**Reprojection Error:** Distribution histogram, mean 0.687 px  
**3D Trajectory:** Camera path with orientations (red=X-axis, blue=Z-axis)  
**Debug Images:** 10 highest-error frames with detected/reprojected corners

## Usage
```bash
# Run detection and PnP estimation
python3 Cam0PnP.py

# Generate visualizations
python3 result_plot.py
```

## Dependencies

- OpenCV (camera calibration, PnP solver)
- aprilgrid (robust tag detection)
- rosbag2_py, mcap (ROS2 bag file reading)
- plotly (interactive 3D visualization)